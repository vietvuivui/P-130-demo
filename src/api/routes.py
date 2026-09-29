import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.config import get_settings
from src.models.qa_config import AutoLabelConfig, get_autolabel_config
from src.models.schemas import (
    ExportResponse,
    FrameRecord,
    FrameSummary,
    PropagateRequest,
    PropagateResponse,
    ReviewActionRequest,
    ReviewerRequest,
    VideoDetail,
    VideoSummary,
)
from src.services import review
from src.services import video as video_service
from src.services.detectors import DetectorEnsemble
from src.services.exporter import EXPORT_FILES, NothingToExportError, export_dataset
from src.services.pipeline import image_path
from src.services.sequence import PropagationError, SequenceSource, WorkspaceSequenceSource, propagate_from
from src.services.store import WorkspaceStore

router = APIRouter()
log = logging.getLogger(__name__)

# Câu giải thích hiển thị cạnh issue code trên UI
ISSUE_HELP = {
    "LOW_CONFIDENCE": "Model không chắc chắn về box này",
    "CLASS_CONFLICT": "Cùng vị trí, model phân vân giữa nhiều lớp",
    "NO_LIDAR_SUPPORT": "Box đủ lớn nhưng gần như không có điểm LiDAR: có thể là phản chiếu, poster, bóng",
    "SIZE_DEPTH_MISMATCH": "Chiều cao suy từ độ sâu LiDAR không hợp với lớp: sai lớp hoặc box sai kích thước",
    "FLICKER": "Box không xuất hiện lại ở các sweep lân cận: dễ là FP ngẫu nhiên",
    "RECOVERED_BY_TRACK": "Detector sót ở keyframe, box nội suy từ sweep trước/sau: cần xác nhận",
    "BOX_TOO_LARGE": "Box chiếm phần lớn ảnh",
    "ASPECT_RATIO_ABNORMAL": "Tỉ lệ rộng/cao bất thường với lớp này",
    "PROP_LOW_CONF": "Nhãn lan truyền từ keyframe trước nhưng độ tin cậy thấp: kiểm tra box có còn đúng object",
    "PROP_COASTING": "Detector không thấy object ở frame này, box chỉ là dự đoán theo chuyển động: dễ lệch hoặc bị che",
    "PROP_CLASS_DIFFERS": "Detector ở frame này nhận lớp khác lớp người đã chốt: có thể track đã nhảy sang object khác",
}


@lru_cache
def _store(root: str) -> WorkspaceStore:
    return WorkspaceStore(root)


def get_store() -> WorkspaceStore:
    return _store(get_settings().workspace_dir)


def get_config() -> AutoLabelConfig:
    return get_autolabel_config(get_settings().autolabel_config)


def get_dataroot() -> Path:
    return Path(get_settings().nuscenes_dataroot)


@lru_cache
def _ensemble(workspace: str, config_path: str) -> DetectorEnsemble:
    # Model chỉ được nạp khi thật sự cần detect (video tải lên); đọc cache thì không cần torch
    return DetectorEnsemble(get_autolabel_config(config_path), Path(workspace) / "cache" / "detections")


def get_ensemble() -> DetectorEnsemble:
    s = get_settings()
    return _ensemble(s.workspace_dir, s.autolabel_config)


@lru_cache
def _nuscenes(dataroot: str, version: str):
    # Bảng nuScenes chỉ đọc một lần cho mọi lần lan truyền (sample_data.json của bản đầy đủ rất lớn).
    # Chưa có dataset thì NuScenesMini ném FileNotFoundError, lru_cache không lưu lỗi nên thêm dataset sau vẫn nhận
    from src.services.nuscenes_data import NuScenesMini

    return NuScenesMini(dataroot, version)


def get_sequence_source(
    store: WorkspaceStore = Depends(get_store), ensemble: DetectorEnsemble = Depends(get_ensemble)
) -> SequenceSource:
    s = get_settings()
    return WorkspaceSequenceSource(store, ensemble, lambda: _nuscenes(s.nuscenes_dataroot, s.nuscenes_version))


def resume_videos() -> None:
    """Gọi khi server khởi động: làm tiếp video tải lên còn dở (server tắt / reload giữa lúc auto-label)."""
    try:
        store = get_store()
        if any(v.status == "processing" for v in store.list_videos()):
            video_service.resume_pending(store, get_ensemble(), get_config())
    except Exception:  # không để lỗi ở đây làm hỏng lúc khởi động server
        log.exception("Không làm tiếp được video đang xử lý")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _load(store: WorkspaceStore, frame_id: str) -> FrameRecord:
    try:
        frame = store.load_frame(frame_id)
    except ValueError as e:
        raise _error(400, "INVALID_FRAME_ID", str(e)) from e
    if frame is None:
        raise _error(404, "FRAME_NOT_FOUND", f"Không có frame {frame_id}")
    return frame


@router.get("/status")
def status(store: WorkspaceStore = Depends(get_store)):
    return {"status": "ready", "frames": len(store.frame_ids()), "agent": "QA Agent (LangGraph, deterministic)"}


@router.get("/config")
def ui_config(config: AutoLabelConfig = Depends(get_config)):
    return {
        "classes": {name: spec.color for name, spec in config.classes.items()},
        "levels": config.qa.risk.levels.model_dump(),
        "weights": config.qa.risk.weights.model_dump(),
        "sweep_offsets": config.sweep_offsets,
        "issue_help": ISSUE_HELP,
        "reviewer": get_settings().reviewer_name,
    }


@router.get("/frames", response_model=list[FrameSummary])
def list_frames(
    status: Literal["auto", "editing", "approved"] | None = None,
    sort: Literal["risk", "order"] = "risk",
    store: WorkspaceStore = Depends(get_store),
):
    """Hàng đợi frame. sort=risk xếp frame khó lên đầu (active learning)."""
    frames = [review.summarize(f) for f in store.list_frames() if status is None or f.status == status]
    if sort == "risk":
        frames.sort(key=lambda s: (s.status == "approved", -s.frame_risk))
    else:
        frames.sort(key=lambda s: (s.scene, s.index))
    return frames


@router.get("/frames/{frame_id}", response_model=FrameRecord)
def get_frame(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    return _load(store, frame_id)


@router.get("/frames/{frame_id}/image")
def get_image(
    frame_id: str,
    offset: int = Query(0, ge=-5, le=5),
    store: WorkspaceStore = Depends(get_store),
    dataroot: Path = Depends(get_dataroot),
):
    frame = _load(store, frame_id)
    try:
        path = image_path(dataroot, frame, offset, workspace=store.root)
    except KeyError as e:
        raise _error(404, "SWEEP_NOT_FOUND", f"Frame không có sweep offset {offset}") from e
    if not path.exists():
        raise _error(404, "IMAGE_NOT_FOUND", "Không tìm thấy file ảnh trong dataroot")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.get("/frames/{frame_id}/lidar")
def get_lidar(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, frame_id)
    return store.load_aux("lidar", frame_id) or {"u": [], "v": [], "d": []}


def _cam_from_ego(frame: FrameRecord) -> tuple[np.ndarray, bool]:
    """Ngoại tham số camera so với hệ ego (mặt đường z = 0). Video tải lên / thiếu bảng: giả định, estimated=True."""
    from src.services import bev

    if not frame.image.path.startswith("@workspace"):
        s = get_settings()
        try:
            data = _nuscenes(s.nuscenes_dataroot, s.nuscenes_version)
            return np.linalg.inv(data._ego_from_sensor(frame.image.sd_token)), False
        except (FileNotFoundError, KeyError):
            pass
    return bev.nominal_cam_from_ego(), True


@router.get("/frames/{frame_id}/bev/meta")
def get_bev_meta(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    """Thông số để UI đặt box 2D / điểm LiDAR lên ảnh BEV: ngoại tham số, homography mặt đường -> ảnh, vùng phủ."""
    from src.services import bev

    frame = _load(store, frame_id)
    cam_from_ego, estimated = _cam_from_ego(frame)
    homo = bev.ground_homography(np.asarray(frame.intrinsic, float), cam_from_ego, 0.0)
    return {
        "cam_from_ego": np.round(cam_from_ego, 6).tolist(), "estimated": estimated,
        "homography": (homo / np.linalg.norm(homo)).round(10).tolist(), "x_range": list(bev.BEV2D_X),
        "y_range": list(bev.BEV2D_Y), "res": 0.1,
        "camera_height": round(float(np.linalg.inv(cam_from_ego)[2, 3]), 3),
    }  # fmt: skip


@router.get("/frames/{frame_id}/bev")
def get_bev(frame_id: str, store: WorkspaceStore = Depends(get_store), dataroot: Path = Depends(get_dataroot)):
    """Ảnh camera của frame chiếu xuống mặt đường (homography), PNG RGBA; phía trước ở trên."""
    from src.services import bev

    frame = _load(store, frame_id)
    cache = store.root / "bev2d" / f"{store.check_id(frame_id)}.png"
    if not cache.exists():
        img = bev.load_rgb(image_path(dataroot, frame, 0, workspace=store.root))
        if img is None:
            raise _error(404, "IMAGE_NOT_FOUND", "Không tìm thấy file ảnh trong dataroot")
        cam_from_ego, _ = _cam_from_ego(frame)
        data = bev.to_png(bev.bev_camera(img, np.asarray(frame.intrinsic, float), cam_from_ego))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    return FileResponse(cache, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.get("/frames/{frame_id}/gt")
def get_gt(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, frame_id)
    return store.load_aux("gt", frame_id) or []


@router.post("/frames/{frame_id}/actions", response_model=FrameRecord)
def review_action(
    frame_id: str,
    req: ReviewActionRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    frame = _load(store, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        entry = review.apply_action(frame, req, reviewer, config)
    except review.ReviewError as e:
        raise _error(422, "INVALID_ACTION", str(e)) from e
    store.save_frame(frame)
    store.append_corrections([entry])
    return frame


@router.post("/frames/{frame_id}/approve-low-risk", response_model=FrameRecord)
def approve_low_risk(frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    try:
        entries = review.approve_low_risk(frame, req.reviewer or get_settings().reviewer_name)
    except review.ReviewError as e:
        raise _error(409, "FRAME_APPROVED", str(e)) from e
    store.save_frame(frame)
    store.append_corrections(entries)
    return frame


@router.post("/frames/{frame_id}/approve", response_model=FrameRecord)
def approve_frame(frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    try:
        review.approve_frame(frame, req.reviewer or get_settings().reviewer_name, req.review_time_s)
    except review.ReviewError as e:
        raise _error(409, "PENDING_OBJECTS", str(e)) from e
    store.save_frame(frame)
    return frame


@router.post("/frames/{frame_id}/propagate", response_model=PropagateResponse)
def propagate(
    frame_id: str,
    req: PropagateRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
    source: SequenceSource = Depends(get_sequence_source),
):
    """Lan truyền quyết định của người ở frame đã approve sang các keyframe sau còn "auto"."""
    _load(store, frame_id)
    try:
        return propagate_from(store, source, config, frame_id, req.max_frames)
    except PropagationError as e:
        raise _error(e.status, e.code, str(e)) from e


# ---- Video ----


@router.get("/videos", response_model=list[VideoSummary])
def list_videos(store: WorkspaceStore = Depends(get_store)):
    """Scene nuScenes (mỗi scene là một video) và video mp4 đã tải lên."""
    return video_service.list_videos(store)


@router.get("/videos/{video_id}", response_model=VideoDetail)
def get_video(video_id: str, store: WorkspaceStore = Depends(get_store)):
    detail = video_service.video_detail(store, video_id)
    if detail is None:
        raise _error(404, "VIDEO_NOT_FOUND", f"Không có video {video_id}")
    return detail


@router.post("/videos/upload", response_model=VideoSummary)
def upload_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    name: str | None = Form(None),
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
    ensemble: DetectorEnsemble = Depends(get_ensemble),
):
    """Tải lên video: cắt frame ngay, auto-label + QA chạy nền (UI hỏi lại GET /videos/{id} để xem tiến độ)."""
    filename = file.filename or "video.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in video_service.VIDEO_EXTENSIONS:
        raise _error(422, "VIDEO_FORMAT", f"Chỉ nhận {', '.join(sorted(video_service.VIDEO_EXTENSIONS))}")
    limit = config.video.max_upload_mb * 1024 * 1024
    too_large = _error(413, "VIDEO_TOO_LARGE", f"Video lớn hơn {config.video.max_upload_mb} MB")
    if file.size is not None and file.size > limit:
        raise too_large
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp:
            copied = 0
            while chunk := file.file.read(1024 * 1024):
                copied += len(chunk)
                if copied > limit:
                    raise too_large
                tmp.write(chunk)
        try:
            video = video_service.import_video(store, config, tmp_path, name or filename, ensemble.names)
        except video_service.VideoError as e:
            raise _error(e.status, e.code, str(e)) from e
        try:  # giữ file gốc để xem lại / xử lý lại; không có cũng không sao
            shutil.copyfile(tmp_path, store.video_dir(video.video_id) / f"source{suffix}")
        except OSError:
            log.warning("Không lưu được file gốc của %s", video.video_id, exc_info=True)
    finally:
        tmp_path.unlink(missing_ok=True)
    background.add_task(video_service.process_video, store, ensemble, config, video.video_id)
    return video_service.video_summary(video.video_id, [], video)


@router.post("/frames/{frame_id}/reopen", response_model=FrameRecord)
def reopen_frame(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    review.reopen_frame(frame)
    store.save_frame(frame)
    return frame


@router.get("/corrections")
def corrections(
    frame_id: str | None = None, limit: int = Query(200, ge=1, le=5000), store: WorkspaceStore = Depends(get_store)
):
    return store.corrections(frame_id)[-limit:][::-1]


@router.get("/metrics")
def metrics(store: WorkspaceStore = Depends(get_store)):
    return review.compute_metrics(store.list_frames())


@router.post("/export", response_model=ExportResponse)
def export(store: WorkspaceStore = Depends(get_store), config: AutoLabelConfig = Depends(get_config)):
    try:
        return export_dataset(store, config)
    except NothingToExportError as e:
        raise _error(409, "NOTHING_TO_EXPORT", str(e)) from e


@router.get("/exports")
def list_exports(store: WorkspaceStore = Depends(get_store)):
    if not store.exports_dir.is_dir():
        return []
    return sorted((p.name for p in store.exports_dir.iterdir() if p.is_dir()), reverse=True)


@router.get("/exports/{export_id}/{filename}")
def download_export(export_id: str, filename: str, store: WorkspaceStore = Depends(get_store)):
    if filename not in EXPORT_FILES:
        raise _error(404, "FILE_NOT_FOUND", filename)
    path = store.exports_dir / WorkspaceStore.check_id(export_id) / filename
    if not path.exists():
        raise _error(404, "FILE_NOT_FOUND", filename)
    return FileResponse(path, filename=f"{export_id}_{filename}")
