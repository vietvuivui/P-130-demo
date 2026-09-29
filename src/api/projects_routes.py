"""API dự án cho end-user: tạo dự án kèm tải dữ liệu lên, theo dõi tiến độ, chạy lại, xoá, xuất kết quả.

Sau khi dự án xử lý xong, UI duyệt dùng chính API duyệt ở /p/{project_id}/api/v1/... (xem src/main.py).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.config import get_settings
from src.models.qa_config import get_autolabel_config
from src.services import ingest
from src.services.projects import MODEL3D, Project, ProjectError, ProjectOptions, get_manager

projects_router = APIRouter(prefix="/projects")


def _err(e: ProjectError | ValueError, status: int = 422, code: str = "INVALID") -> HTTPException:
    if isinstance(e, ProjectError):
        status, code = e.status, e.code
    return HTTPException(status_code=status, detail={"code": code, "message": str(e)})


def _get(pid: str) -> Project:
    try:
        return get_manager().get(pid)
    except ProjectError as e:
        raise _err(e) from e


@projects_router.get("", response_model=list[Project])
def list_projects():
    return get_manager().list()


@projects_router.post("", response_model=Project)
def create_project(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    kind: str = Form("auto"),
    tta: bool = Form(False),
    sequential: bool = Form(False),
    max_frames: int | None = Form(None),
):
    """Tải lên: một video, nhiều ảnh, hoặc một file nén (.zip/.tar) chứa nuScenes / KITTI / LiDAR + camera / ảnh."""
    if len(files) > 1 and not all(Path(f.filename or "").suffix.lower() in ingest.IMAGE_EXT for f in files):
        raise HTTPException(
            422, {"code": "MULTI_FILES", "message": "Nhiều file chỉ nhận ảnh; dữ liệu khác hãy nén thành một file .zip"}
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="upload-"))
    saved = []
    try:
        for f in files:
            dest = tmp_dir / Path(f.filename or "upload.bin").name
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out, 1 << 20)
            saved.append(dest)
        ok_ext = ingest.VIDEO_EXT | ingest.IMAGE_EXT
        for p in saved:
            if not (ingest.is_archive(p.name) or p.suffix.lower() in ok_ext):
                raise HTTPException(422, {"code": "FILE_TYPE", "message": f"Không nhận loại file {p.name}"})
        m = get_manager()
        opts = ProjectOptions(tta=tta, sequential=sequential, max_frames=max_frames or None)
        project = m.create(name or Path(saved[0].name).stem, saved, kind, opts)
        return m.enqueue(project.id)
    except ProjectError as e:
        raise _err(e) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@projects_router.get("/{pid}", response_model=Project)
def get_project(pid: str):
    return _get(pid)


@projects_router.post("/{pid}/run", response_model=Project)
def rerun_project(pid: str):
    """Chạy lại: bước lỗi / chưa xong được làm tiếp, bước đã xong giữ nguyên."""
    p = _get(pid)
    if p.status == "processing":
        raise HTTPException(409, {"code": "PROJECT_BUSY", "message": "Dự án đang xử lý"})
    for s in p.steps:
        if s.status == "error":
            s.status, s.message = "pending", None
    get_manager().save(p)
    return get_manager().enqueue(pid)


@projects_router.delete("/{pid}")
def delete_project(pid: str):
    try:
        get_manager().delete(pid)
    except ProjectError as e:
        raise _err(e) from e
    return {"deleted": pid}


@projects_router.get("/{pid}/log/{step}")
def project_log(pid: str, step: str, lines: int = Query(80, ge=1, le=2000)):
    _get(pid)
    path = get_manager().dir(pid) / "logs" / f"{Path(step).name}.log"
    if not path.exists():
        return {"lines": []}
    return {"lines": path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]}


@projects_router.post("/{pid}/export")
def export_project(pid: str, fmt: str = Query("nuscenes", alias="format"), include_pending: bool = False):
    """format=nuscenes: nhãn 3D thành dataset nuScenes (zip). format=coco: nhãn 2D (COCO + JSONL)."""
    p = _get(pid)
    m = get_manager()
    store = m.store(pid)
    exports = m.dir(pid) / "exports"
    if fmt == "nuscenes":
        if not p.stats.get("has_lidar"):
            raise HTTPException(409, {"code": "NO_3D", "message": "Dự án chỉ có ảnh / video (2D): dùng xuất COCO"})
        from src.services.export_nusc import export_nuscenes
        from src.services.review import now_iso

        root, version = m.dataset(pid)
        name = f"nuscenes-{now_iso().replace(':', '').replace('-', '')[:15]}.zip"
        try:
            info = export_nuscenes(store, root, version, MODEL3D, exports / name, include_pending=include_pending)
        except ValueError as e:
            raise HTTPException(409, {"code": "NOTHING_TO_EXPORT", "message": str(e)}) from e
        return dict(info, url=f"/api/v1/projects/{pid}/exports/{name}")
    if fmt == "coco":
        from src.services.exporter import NothingToExportError, export_dataset

        try:
            res = export_dataset(store, get_autolabel_config(get_settings().autolabel_config))
        except NothingToExportError as e:
            raise HTTPException(409, {"code": "NOTHING_TO_EXPORT", "message": str(e)}) from e
        src_dir = store.exports_dir / res.export_id
        name = f"coco-{res.export_id}.zip"
        exports.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(exports / name[:-4]), "zip", src_dir)
        return dict(file=name, frames=res.n_frames, objects=res.n_objects, url=f"/api/v1/projects/{pid}/exports/{name}")
    raise HTTPException(422, {"code": "FORMAT", "message": "format phải là nuscenes hoặc coco"})


@projects_router.get("/{pid}/exports")
def list_project_exports(pid: str):
    _get(pid)
    d = get_manager().dir(pid) / "exports"
    files = sorted(d.glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True) if d.exists() else []
    return [dict(file=f.name, size=f.stat().st_size, url=f"/api/v1/projects/{pid}/exports/{f.name}") for f in files]


@projects_router.get("/{pid}/exports/{name}")
def download_project_export(pid: str, name: str):
    _get(pid)
    path = get_manager().dir(pid) / "exports" / Path(name).name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(404, {"code": "FILE_NOT_FOUND", "message": name})
    return FileResponse(path, filename=path.name, media_type="application/zip")
