"""Dự án của end-user: nhận dữ liệu, xử lý nền (đổi định dạng -> nhãn 2D -> dự đoán 3D -> kiểm chứng 3D), theo dõi tiến độ.

Mỗi dự án một thư mục data/projects/<id>/:
    project.json    trạng thái + tiến độ từng bước
    upload/         file người dùng tải lên (giữ nguyên)
    dataset/        dữ liệu đã đổi sang nuScenes (v1.0-custom/ hoặc bảng gốc của zip nuScenes) — dự án có LiDAR
    workspace/      frame, nhãn, log sửa, export (WorkspaceStore) — UI duyệt đọc ở đây
    work3d/         file info / dự đoán của tools3d/run3d.py (môi trường MMDetection3D riêng)
    logs/           log từng bước

Các bước chạy lần lượt trên một luồng nền (một GPU, detector không an toàn khi chạy song song). Bước nào cũng làm
tiếp được: chạy lại dự án chỉ làm phần còn thiếu.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.models.qa_config import AutoLabelConfig
from src.services import ingest
from src.services.store import WorkspaceStore

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
StepStatus = Literal["pending", "running", "done", "skipped", "error"]
STEP_LABEL = {
    "ingest": "Nhận dữ liệu",
    "label2d": "Gán nhãn 2D + QA",
    "predict3d": "Dự đoán 3D (ensemble)",
    "verify3d": "Kiểm chứng 3D bằng camera",
}
STEPS = {
    "video": ["ingest", "label2d"],
    "images": ["ingest", "label2d"],
    "nuscenes": ["ingest", "label2d", "predict3d", "verify3d"],
    "kitti": ["ingest", "label2d", "predict3d", "verify3d"],
    "lidar_cam": ["ingest", "label2d", "predict3d", "verify3d"],
}
MODEL3D = "ensemble"  # tên "mô hình" 3D của dự án trong workspace (frames3d/ensemble)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ProjectStep(BaseModel):
    name: str
    label: str
    status: StepStatus = "pending"
    progress: float = 0.0
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ProjectOptions(BaseModel):
    tta: bool = False  # 3D: thêm 3 lượt lật trục (chậm ~4 lần, chính xác hơn)
    sequential: bool = False  # bộ ảnh: True = frame liên tiếp ~10 fps (gán nhãn 1/5 ảnh, còn lại lan truyền)
    max_frames: int | None = None  # giới hạn số frame (thử nhanh)


class Project(BaseModel):
    id: str
    name: str
    kind: str | None = None  # None: chưa nhận dạng (tự nhận khi xử lý)
    status: Literal["queued", "processing", "ready", "error"] = "queued"
    message: str | None = None
    created_at: str
    updated_at: str | None = None
    options: ProjectOptions = Field(default_factory=ProjectOptions)
    uploads: list[str] = Field(default_factory=list)
    steps: list[ProjectStep] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)  # frames, scenes, cameras, has_lidar, version, video_id...


class ProjectError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:24]
    return f"p-{s or 'du-an'}-{uuid.uuid4().hex[:6]}"


class ProjectManager:
    def __init__(self, root: str | Path, config_loader, mm3d_python: str = "", models3d: list[str] | None = None,
                 predict_args: list[str] | None = None):  # fmt: skip
        self.root = Path(root).resolve()  # tiến trình 3D chạy với cwd = gốc repo: đường dẫn phải tuyệt đối
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_loader = config_loader  # () -> AutoLabelConfig
        self.mm3d_python = mm3d_python
        self.models3d = models3d  # None: 4 mô hình LiDAR của ensemble (run3d.py ENSEMBLE_MODELS)
        self.predict_args = predict_args or []
        self._lock = threading.RLock()
        self._stores: dict[str, WorkspaceStore] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._queued: set[str] = set()

    # ---------------------------------------------------------------- đường dẫn
    def dir(self, pid: str) -> Path:
        if not re.fullmatch(r"p-[a-z0-9-]{1,40}", pid or ""):
            raise ProjectError("INVALID_PROJECT_ID", f"Mã dự án không hợp lệ: {pid}", 400)
        return self.root / pid

    def store(self, pid: str) -> WorkspaceStore:
        with self._lock:
            if pid not in self._stores:
                self._stores[pid] = WorkspaceStore(self.dir(pid) / "workspace")
            return self._stores[pid]

    def dataset(self, pid: str) -> tuple[Path, str]:
        p = self.get(pid)
        ds = self.dir(pid) / "dataset"
        return Path(p.stats.get("dataroot") or ds), p.stats.get("version", "v1.0-custom")

    # ---------------------------------------------------------------- CRUD
    def _path(self, pid: str) -> Path:
        return self.dir(pid) / "project.json"

    def get(self, pid: str) -> Project:
        path = self._path(pid)
        if not path.exists():
            raise ProjectError("PROJECT_NOT_FOUND", f"Không có dự án {pid}", 404)
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, p: Project) -> None:
        with self._lock:
            p.updated_at = now_iso()
            path = self._path(p.id)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(p.model_dump_json(indent=1), encoding="utf-8")
            os.replace(tmp, path)

    def list(self) -> list[Project]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if (d / "project.json").exists():
                try:
                    out.append(Project.model_validate_json((d / "project.json").read_text(encoding="utf-8")))
                except ValueError:
                    log.warning("Bỏ qua dự án hỏng %s", d.name)
        return sorted(out, key=lambda p: p.created_at, reverse=True)

    def create(self, name: str, uploads: list[Path], kind: str | None = None,
               options: ProjectOptions | None = None) -> Project:  # fmt: skip
        """uploads: file đã lưu tạm; được chuyển vào thư mục dự án."""
        if kind not in (None, "auto", *ingest.KINDS):
            raise ProjectError("INVALID_KIND", f"Loại dữ liệu không hợp lệ: {kind}")
        if not uploads:
            raise ProjectError("NO_FILES", "Chưa chọn file nào")
        pid = slug(name)
        up = self.dir(pid) / "upload"
        up.mkdir(parents=True, exist_ok=True)
        names = []
        for f in uploads:
            dest = up / Path(f.name).name
            shutil.move(str(f), dest)
            names.append(dest.name)
        p = Project(id=pid, name=name.strip() or pid, kind=None if kind in (None, "auto") else kind,
                    created_at=now_iso(), options=options or ProjectOptions(), uploads=names)  # fmt: skip
        self.save(p)
        return p

    def delete(self, pid: str) -> None:
        p = self.get(pid)
        if p.status == "processing":
            raise ProjectError("PROJECT_BUSY", "Dự án đang xử lý, chờ xong rồi xoá", 409)
        with self._lock:
            self._stores.pop(pid, None)
        shutil.rmtree(self.dir(pid), ignore_errors=True)

    # ---------------------------------------------------------------- hàng đợi
    def enqueue(self, pid: str) -> Project:
        p = self.get(pid)
        with self._lock:
            if pid in self._queued:
                return p
            self._queued.add(pid)
            p.status, p.message = "queued", "Đang chờ tới lượt xử lý"
            self.save(p)
            self._queue.put(pid)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="project-worker", daemon=True)
                self._worker.start()
        return p

    def resume(self) -> list[str]:
        """Khi server khởi động: dự án đang chờ / đang xử lý dở được xếp hàng lại (các bước làm tiếp được)."""
        pending = [p.id for p in self.list() if p.status in ("queued", "processing")]
        for pid in pending:
            self.enqueue(pid)
        return pending

    def _work(self) -> None:
        while True:
            pid = self._queue.get()
            try:
                self.run(pid)
            except Exception:  # không để lỗi của một dự án làm chết luồng nền
                log.exception("Dự án %s lỗi", pid)
            finally:
                with self._lock:
                    self._queued.discard(pid)
                self._queue.task_done()

    def wait_idle(self, timeout: float = 600) -> None:
        """Chờ hàng đợi trống (dùng trong test)."""
        t0 = time.time()
        while (self._queued or not self._queue.empty()) and time.time() - t0 < timeout:
            time.sleep(0.05)

    # ---------------------------------------------------------------- chạy các bước
    def _step(self, p: Project, name: str) -> ProjectStep:
        return next(s for s in p.steps if s.name == name)

    def _set(self, p: Project, name: str, **kw) -> None:
        st = self._step(p, name)
        for k, v in kw.items():
            setattr(st, k, v)
        self.save(p)

    def run(self, pid: str) -> Project:
        from src.services.video import _PROCESS_LOCK

        p = self.get(pid)
        p.status, p.message = "processing", None
        if p.kind is None:
            try:
                p.kind = self._detect_kind(p)
            except ingest.IngestError as e:
                p.status, p.message = "error", str(e)
                self.save(p)
                return p
        names = STEPS[p.kind]
        old = {s.name: s for s in p.steps}
        p.steps = [old.get(n) or ProjectStep(name=n, label=STEP_LABEL[n]) for n in names]
        self.save(p)
        config = self.config_loader()
        for name in names:
            st = self._step(p, name)
            if st.status in ("done", "skipped"):
                continue
            self._set(p, name, status="running", progress=0.0, message=None, started_at=now_iso(), finished_at=None)
            try:
                with _PROCESS_LOCK:  # dùng chung khoá GPU với video tải lên ở chế độ cũ
                    result = getattr(self, f"_do_{name}")(p, config)
                status, msg = ("skipped", result) if isinstance(result, str) else ("done", None)
                self._set(p, name, status=status, progress=1.0, message=msg, finished_at=now_iso())
            except Exception as e:
                log.exception("Dự án %s, bước %s lỗi", pid, name)
                self._set(p, name, status="error", message=f"{type(e).__name__}: {e}", finished_at=now_iso())
                p.status, p.message = "error", f"{STEP_LABEL[name]}: {e}"
                self.save(p)
                return p
        p.status, p.message = "ready", None
        self.save(p)
        return p

    def _progress(self, p: Project, name: str):
        last = [0.0]

        def cb(done: int, total: int) -> None:
            now = time.time()
            if now - last[0] < 0.5 and done < total:  # không ghi project.json quá dày
                return
            last[0] = now
            self._set(p, name, progress=round(done / max(total, 1), 3), message=f"{done}/{total}")

        return cb

    def _detect_kind(self, p: Project) -> str:
        up = self.dir(p.id) / "upload"
        files = [up / n for n in p.uploads]
        if len(files) == 1 and ingest.is_archive(files[0].name):
            tmp = self.dir(p.id) / "extracted"
            if not tmp.exists():
                ingest.safe_extract(files[0], tmp)
            return ingest.detect_kind(tmp)
        return ingest.detect_kind(up)

    def _source_dir(self, p: Project) -> Path:
        up = self.dir(p.id) / "upload"
        files = [up / n for n in p.uploads]
        if len(files) == 1 and ingest.is_archive(files[0].name):
            tmp = self.dir(p.id) / "extracted"
            if not tmp.exists():
                ingest.safe_extract(files[0], tmp)
            return tmp
        return up

    # ---- bước 1: nhận dữ liệu
    def _do_ingest(self, p: Project, config: AutoLabelConfig):
        from src.services import video as video_service

        store = self.store(p.id)
        src = self._source_dir(p)
        ds = self.dir(p.id) / "dataset"
        prog = self._progress(p, "ingest")
        if p.kind == "video":
            vid = next(f for f in sorted(src.rglob("*")) if f.suffix.lower() in ingest.VIDEO_EXT)
            rec = video_service.import_video(store, config, vid, vid.name, config.detection.detectors)
            p.stats.update(video_id=rec.video_id, frames=len(rec.timeline), has_lidar=False,
                           keyframes=sum(1 for e in rec.timeline if e.sample_token))  # fmt: skip
        elif p.kind == "images":
            files = ingest.image_files(src)
            if p.options.max_frames:
                files = files[: p.options.max_frames]
            rec = video_service.import_images(store, config, files, p.name, config.detection.detectors,
                                              sequential=p.options.sequential)  # fmt: skip
            p.stats.update(video_id=rec.video_id, frames=len(rec.timeline), has_lidar=False,
                           keyframes=sum(1 for e in rec.timeline if e.sample_token))  # fmt: skip
        elif p.kind == "nuscenes":
            found = ingest.find_nuscenes(src)
            if not found:
                raise ingest.IngestError("Không thấy bảng nuScenes (thư mục v1.0-*/ có scene.json)")
            root, version = found
            if ds.exists():
                shutil.rmtree(ds)
            shutil.move(str(root), ds)  # dataset/ = dataroot (samples/, sweeps/, v1.0-*/)
            p.stats.update(version=version, has_lidar=True, has_poses=True, **self._count(ds, version))
        else:
            from src.services.ingest import kitti, lidarcam

            if ds.exists():
                shutil.rmtree(ds)
            conv = kitti.convert if p.kind == "kitti" else lidarcam.convert
            info = conv(src, ds, "v1.0-custom", max_frames=p.options.max_frames,
                        progress=lambda n: self._set(p, "ingest", message=f"{n} frame"))  # fmt: skip
            p.stats.update(info)
            p.stats.setdefault("has_poses", p.kind == "kitti")
        extracted = self.dir(p.id) / "extracted"
        if extracted.exists() and p.kind in ("nuscenes", "kitti", "lidar_cam"):
            shutil.rmtree(extracted, ignore_errors=True)  # đã chuyển / đổi xong, bỏ bản giải nén
        prog(1, 1)
        self.save(p)

    @staticmethod
    def _count(ds: Path, version: str) -> dict:
        t = ds / version
        scenes = json.loads((t / "scene.json").read_text())
        samples = json.loads((t / "sample.json").read_text())
        return dict(scenes=len(scenes), frames=len(samples))

    # ---- bước 2: nhãn 2D
    def _do_label2d(self, p: Project, config: AutoLabelConfig):
        from src.services import video as video_service

        store = self.store(p.id)
        prog = self._progress(p, "label2d")
        if p.kind in ("video", "images"):
            from src.services.detectors import DetectorEnsemble

            ens = DetectorEnsemble(config, store.root / "cache" / "detections")
            vid = p.stats["video_id"]
            video_service._process_video(store, ens, config, vid)  # đã giữ khoá GPU ở run()
            rec = store.load_video(vid)
            if rec is not None and rec.status == "error":
                raise RuntimeError(rec.message)
            return None
        from src.services.nuscenes_data import NuScenesMini
        from src.services.pipeline import AutoLabelPipeline

        root, version = self.dataset(p.id)
        data = NuScenesMini(root, version)
        pipe = AutoLabelPipeline(data, store, config)
        done = pipe.run(limit=p.options.max_frames, progress=prog)
        p.stats["frames2d"] = len(store.frame_ids())
        self.save(p)
        return None if done or store.frame_ids() else "Không có keyframe có CAM_FRONT"

    # ---- bước 3: dự đoán 3D (môi trường MMDetection3D riêng)
    def mm3d(self) -> Path | None:
        if self.mm3d_python:
            py = Path(self.mm3d_python).absolute()  # absolute, không resolve: python của venv là symlink
            return py if py.exists() else None
        for cand in (ROOT / ".venv-mm3d" / "Scripts" / "python.exe", ROOT / ".venv-mm3d" / "bin" / "python"):
            if cand.exists():
                return cand
        return None

    def _do_predict3d(self, p: Project, config: AutoLabelConfig):
        py = self.mm3d()
        if py is None:
            return "Chưa cài môi trường 3D (.venv-mm3d, xem tools3d/README.md): dự án chỉ có nhãn 2D"
        root, version = self.dataset(p.id)
        work = self.dir(p.id) / "work3d"
        out = work / "preds.json"
        if out.exists():
            return None
        cmd = [str(py), str(ROOT / "tools3d" / "run3d.py"), "predict", "--dataroot", str(root), "--version", version,
               "--work", str(work), "--out", str(out)]  # fmt: skip
        if self.models3d:
            cmd += ["-m", *self.models3d]
        if p.options.tta:
            cmd.append("--tta")
        if not p.stats.get("has_poses", True):
            cmd += ["--max-sweeps", "0"]  # không có tư thế xe: gộp các lần quét trước sẽ bị nhoè khi xe chạy
        cmd += self.predict_args
        n_runs = len(self.models3d or [1, 2, 3, 4]) * (4 if p.options.tta else 1)
        logs = self.dir(p.id) / "logs"
        logs.mkdir(exist_ok=True)
        done_runs, tail = 0, []
        with open(logs / "predict3d.log", "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))  # fmt: skip
            for line in proc.stdout:
                lf.write(line)
                tail = (tail + [line.rstrip()])[-15:]
                m = re.search(r"Epoch\(test\)\s*\[\s*(\d+)/(\d+)\]", line)
                if m:
                    frac = (done_runs + int(m.group(1)) / int(m.group(2))) / n_runs
                    self._set(p, "predict3d", progress=round(min(frac, 0.99), 3))
                elif re.match(r"^\S+: \d+s \(", line):
                    done_runs += 1
            code = proc.wait()
        if code != 0 or not out.exists():
            raise RuntimeError("run3d.py predict lỗi:\n" + "\n".join(tail[-8:]))
        return None

    # ---- bước 4: kiểm chứng 3D bằng camera, tạo frame 3D cho UI
    def _do_verify3d(self, p: Project, config: AutoLabelConfig):
        from src.services.label3d import auto_min_score, run_label3d
        from src.services.nuscenes_data import NuScenesMini

        preds_file = self.dir(p.id) / "work3d" / "preds.json"
        if not preds_file.exists():
            return "Không có dự đoán 3D"
        preds = json.loads(preds_file.read_text())["results"]
        root, version = self.dataset(p.id)
        data = NuScenesMini(root, version)
        min_score = auto_min_score(MODEL3D, config.verify3d.min_score, ROOT / "eval" / "results" / "det3d")
        n = run_label3d(self.store(p.id), data, config, MODEL3D, preds, min_score=min_score,
                        progress=self._progress(p, "verify3d"))  # fmt: skip
        p.stats["frames3d"] = len(self.store(p.id).frame3d_ids(MODEL3D))
        p.stats["min_score3d"] = min_score
        self.save(p)
        return None if n or p.stats["frames3d"] else "Không có keyframe nào có dự đoán 3D"


_manager: ProjectManager | None = None


def get_manager() -> ProjectManager:
    global _manager
    if _manager is None:
        from src.config import get_settings
        from src.models.qa_config import get_autolabel_config

        s = get_settings()
        _manager = ProjectManager(s.projects_dir, lambda: get_autolabel_config(s.autolabel_config), s.mm3d_python)
    return _manager


if __name__ == "__main__":  # pragma: no cover - tiện chạy tay: python -m src.services.projects <id>
    logging.basicConfig(level=logging.INFO)
    print(get_manager().run(sys.argv[1]).model_dump_json(indent=1))
