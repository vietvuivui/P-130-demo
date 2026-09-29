"""API phần 3D: danh sách frame 3D theo mô hình, point cloud, ảnh 6 camera, thao tác duyệt, số liệu, so sánh mô hình."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response

from src.api.routes import _error, get_config, get_dataroot, get_store
from src.config import get_settings
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import ReviewerRequest
from src.models.schemas3d import Action3DRequest, Frame3DRecord, Frame3DSummary
from src.services import bev, review3d
from src.services.label3d import summarize3d
from src.services.store import WorkspaceStore

router3d = APIRouter(prefix="/3d")
DET3D_DIR = Path("eval/results/det3d")


def _load(store: WorkspaceStore, model: str, frame_id: str) -> Frame3DRecord:
    try:
        frame = store.load_frame3d(model, frame_id)
    except ValueError as e:
        raise _error(400, "INVALID_ID", str(e)) from e
    if frame is None:
        raise _error(404, "FRAME_NOT_FOUND", f"Không có frame 3D {frame_id} của {model}")
    return frame


@router3d.get("/models")
def models(store: WorkspaceStore = Depends(get_store)) -> list[dict]:
    """Mô hình 3D đã có frame trong workspace, kèm số liệu so sánh (nếu đã chạy tools3d/run3d.py eval)."""
    summary = _det3d_summary().get("models", {})
    out = []
    for m in store.models3d():
        frames = store.list_frames3d(m)
        s = summary.get(m, {})
        out.append({
            "model": m, "label": s.get("label", m), "sensor": s.get("sensor"), "frames": len(frames),
            "approved": sum(f.status == "approved" for f in frames), "mAP": s.get("mAP"), "NDS": s.get("NDS"),
        })  # fmt: skip
    # mô hình tốt nhất (NDS cao nhất) đứng đầu: UI mở nó mặc định
    return sorted(out, key=lambda r: (-(r["NDS"] or 0), r["model"]))


@router3d.get("/frames", response_model=list[Frame3DSummary])
def list_frames(model: str, store: WorkspaceStore = Depends(get_store)):
    try:
        return [summarize3d(f) for f in store.list_frames3d(model)]
    except ValueError as e:
        raise _error(400, "INVALID_ID", str(e)) from e


@router3d.get("/frames/{model}/{frame_id}", response_model=Frame3DRecord)
def get_frame(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    return _load(store, model, frame_id)


@router3d.get("/frames/{model}/{frame_id}/points")
def points(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    """Point cloud rút gọn: float32 liên tiếp x, y, z, intensity (hệ LiDAR của keyframe)."""
    _load(store, model, frame_id)
    path = store.points_path(frame_id)
    if not path.exists():
        raise _error(404, "NO_POINTS", "Chưa có point cloud của frame này")
    return Response(path.read_bytes(), media_type="application/octet-stream")


@router3d.get("/frames/{model}/{frame_id}/gt")
def gt(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, model, frame_id)
    return store.load_aux("gt3d", frame_id) or []


@router3d.get("/frames/{model}/{frame_id}/image/{camera}")
def image(model: str, frame_id: str, camera: str, store: WorkspaceStore = Depends(get_store),
          dataroot: Path = Depends(get_dataroot)):  # fmt: skip
    frame = _load(store, model, frame_id)
    cam = frame.cameras.get(camera)
    if cam is None:
        raise _error(404, "NO_CAMERA", f"Frame không có {camera}")
    path = dataroot / cam.path
    if not path.is_file():
        raise _error(404, "IMAGE_NOT_FOUND", f"Không thấy ảnh {cam.path} (kiểm tra NUSCENES_DATAROOT)")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router3d.get("/frames/{model}/{frame_id}/bev")
def bev_image(model: str, frame_id: str, range_m: float = Query(40.0, alias="range", ge=10, le=80),
              res: float = Query(0.1, ge=0.05, le=0.5), store: WorkspaceStore = Depends(get_store),
              dataroot: Path = Depends(get_dataroot)):  # fmt: skip
    """Ảnh BEV ghép 6 camera bằng homography mặt đường (PNG RGBA). Header X-Ground-Z: độ cao mặt đường (hệ LiDAR)."""
    frame = _load(store, model, frame_id)
    pts_path = store.points_path(frame_id)
    pts = np.fromfile(pts_path, np.float32).reshape(-1, 4) if pts_path.exists() else None
    z0 = bev.ground_height(pts)
    # camera và mặt đường như nhau giữa các mô hình: cache theo frame
    cache = store.root / "bev3d" / f"{store.check_id(frame_id)}_{range_m:g}_{res:g}.png"
    if cache.exists():
        data = cache.read_bytes()
    else:
        images = {c: bev.load_rgb(dataroot / cam.path) for c, cam in frame.cameras.items()}
        if not any(v is not None for v in images.values()):
            raise _error(404, "IMAGE_NOT_FOUND", "Không đọc được ảnh camera nào (kiểm tra NUSCENES_DATAROOT)")
        data = bev.to_png(bev.bev_mosaic(images, frame, z0, range_m, res))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    headers = {"X-Ground-Z": str(z0), "X-BEV-Range": f"{range_m:g}", "Cache-Control": "max-age=3600"}
    return Response(data, media_type="image/png", headers=headers)


@router3d.post("/frames/{model}/{frame_id}/actions", response_model=Frame3DRecord)
def action(model: str, frame_id: str, req: Action3DRequest, store: WorkspaceStore = Depends(get_store),
           config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    frame = _load(store, model, frame_id)
    try:
        entry = review3d.apply_action(frame, req, req.reviewer or get_settings().reviewer_name, list(config.classes))
    except review3d.Review3DError as e:
        raise _error(422, "INVALID_ACTION", str(e)) from e
    store.save_frame3d(frame)
    review3d.append_log(store, [entry])
    return frame


@router3d.post("/frames/{model}/{frame_id}/approve-low-risk", response_model=Frame3DRecord)
def approve_low(model: str, frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    try:
        entries = review3d.approve_low_risk(frame, req.reviewer or get_settings().reviewer_name)
    except review3d.Review3DError as e:
        raise _error(409, "FRAME_APPROVED", str(e)) from e
    store.save_frame3d(frame)
    review3d.append_log(store, entries)
    return frame


@router3d.post("/frames/{model}/{frame_id}/approve", response_model=Frame3DRecord)
def approve(model: str, frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    try:
        review3d.approve_frame(frame, req.reviewer or get_settings().reviewer_name, req.review_time_s)
    except review3d.Review3DError as e:
        raise _error(409, "PENDING_OBJECTS", str(e)) from e
    store.save_frame3d(frame)
    return frame


@router3d.post("/frames/{model}/{frame_id}/reopen", response_model=Frame3DRecord)
def reopen(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    review3d.reopen(frame)
    store.save_frame3d(frame)
    return frame


@router3d.get("/metrics")
def metrics(model: str, store: WorkspaceStore = Depends(get_store)):
    return review3d.metrics3d(store.list_frames3d(model))


@router3d.get("/corrections")
def corrections(model: str | None = None, limit: int = Query(500, le=5000), store: WorkspaceStore = Depends(get_store)):
    return review3d.read_log(store, model)[-limit:]


@router3d.post("/export")
def export(model: str, store: WorkspaceStore = Depends(get_store)):
    from src.services.label3d import now_iso

    out = store.exports_dir / f"3d-{model}-{now_iso().replace(':', '').replace('-', '')[:15]}"
    try:
        return review3d.export3d(store, model, out)
    except review3d.Review3DError as e:
        raise _error(409, "NOTHING_TO_EXPORT", str(e)) from e


def _det3d_summary() -> dict:
    path = DET3D_DIR / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router3d.get("/compare")
def compare():
    """Bảng so sánh mô hình 3D (mAP/NDS chuẩn nuScenes trên scene val, tốc độ, kết quả kiểm chứng)."""
    out = _det3d_summary()
    for name in out.get("models", {}):
        v = DET3D_DIR / name / "verify_eval.json"
        if v.exists():
            out["models"][name]["verify"] = json.loads(v.read_text(encoding="utf-8"))
    return out
