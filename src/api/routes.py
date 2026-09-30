import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from src.config import get_settings
from src.models.qa_config import AutoLabelConfig, get_autolabel_config
from src.models.schemas import (
    ExportResponse,
    FrameRecord,
    FrameSummary,
    PropagateRequest,
    PropagateResponse,
    RejectRequest,
    ReviewActionRequest,
    ReviewerRequest,
    SweepActionRequest,
    VideoDetail,
    VideoSummary,
)
from src.services import history, review
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


def _project_id(request: Request | None) -> str | None:
    """API được gắn hai lần: /api/v1 (workspace mặc định trong .env) và /p/{project_id}/api/v1 (dự án của end-user)."""
    return request.path_params.get("project_id") if request is not None else None


def get_store(request: Request = None) -> WorkspaceStore:
    pid = _project_id(request)
    if pid:
        from src.services.projects import ProjectError, get_manager

        try:
            get_manager().get(pid)
            return get_manager().store(pid)
        except ProjectError as e:
            raise _error(e.status, e.code, str(e)) from e
    return _store(get_settings().workspace_dir)


def get_config() -> AutoLabelConfig:
    return get_autolabel_config(get_settings().autolabel_config)


def data_source(request: Request | None) -> tuple[str, str]:
    """(dataroot, version) nuScenes của workspace mặc định hoặc của dự án."""
    pid = _project_id(request)
    if pid:
        from src.services.projects import get_manager

        root, version = get_manager().dataset(pid)
        return str(root), version
    s = get_settings()
    return s.nuscenes_dataroot, s.nuscenes_version


def get_dataroot(request: Request = None) -> Path:
    return Path(data_source(request)[0])


@lru_cache
def _ensemble(workspace: str, config_path: str) -> DetectorEnsemble:
    # Model chỉ được nạp khi thật sự cần detect (video tải lên); đọc cache thì không cần torch
    return DetectorEnsemble(get_autolabel_config(config_path), Path(workspace) / "cache" / "detections")


def get_ensemble(request: Request = None) -> DetectorEnsemble:
    s = get_settings()
    pid = _project_id(request)
    return _ensemble(str(get_store(request).root) if pid else s.workspace_dir, s.autolabel_config)


@lru_cache
def _nuscenes(dataroot: str, version: str):
    # Bảng nuScenes chỉ đọc một lần cho mọi lần lan truyền (sample_data.json của bản đầy đủ rất lớn).
    # Chưa có dataset thì NuScenesMini ném FileNotFoundError, lru_cache không lưu lỗi nên thêm dataset sau vẫn nhận
    from src.services.nuscenes_data import NuScenesMini

    return NuScenesMini(dataroot, version)


def get_sequence_source(
    request: Request,
    store: WorkspaceStore = Depends(get_store),
    ensemble: DetectorEnsemble = Depends(get_ensemble),
) -> SequenceSource:
    root, version = data_source(request)
    return WorkspaceSequenceSource(store, ensemble, lambda: _nuscenes(root, version), root)


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
    status: Literal["auto", "editing", "approved", "rejected"] | None = None,
    sort: Literal["risk", "order"] = "risk",
    store: WorkspaceStore = Depends(get_store),
):
    """Hàng đợi frame. sort=risk xếp frame khó lên đầu (active learning)."""
    frames = [review.summarize(f) for f in store.list_frames() if status is None or f.status == status]
    if sort == "risk":
        # frame bị reviewer trả lại lên đầu, frame đã duyệt xuống cuối
        frames.sort(key=lambda s: (s.status == "approved", s.status != "rejected", -s.frame_risk))
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
    config: AutoLabelConfig = Depends(get_config),
):
    frame = _load(store, frame_id)
    try:
        path = image_path(dataroot, frame, offset, workspace=store.root)
    except KeyError as e:
        raise _error(404, "SWEEP_NOT_FOUND", f"Frame không có sweep offset {offset}") from e
    if not path.exists():
        raise _error(404, "IMAGE_NOT_FOUND", "Không tìm thấy file ảnh trong dataroot")
    from src.services.privacy import anonymized_path

    path = anonymized_path(store.root, path, config.privacy)  # làm mờ mặt / biển số trước khi gửi (FR-03)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.get("/frames/{frame_id}/lidar")
def get_lidar(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, frame_id)
    return store.load_aux("lidar", frame_id) or {"u": [], "v": [], "d": []}


def _cam_from_ego(frame: FrameRecord, request: Request | None = None) -> tuple[np.ndarray, bool]:
    """Ngoại tham số camera so với hệ ego (mặt đường z = 0). Video tải lên / thiếu bảng: giả định, estimated=True."""
    from src.services import bev

    if not frame.image.path.startswith("@workspace"):
        root, version = data_source(request)
        try:
            data = _nuscenes(root, version)
            return np.linalg.inv(data._ego_from_sensor(frame.image.sd_token)), False
        except (FileNotFoundError, KeyError):
            pass
    return bev.nominal_cam_from_ego(), True


@router.get("/frames/{frame_id}/bev/meta")
def get_bev_meta(frame_id: str, request: Request, store: WorkspaceStore = Depends(get_store)):
    """Thông số để UI đặt box 2D / điểm LiDAR lên ảnh BEV: ngoại tham số, homography mặt đường -> ảnh, vùng phủ."""
    from src.services import bev

    frame = _load(store, frame_id)
    cam_from_ego, estimated = _cam_from_ego(frame, request)
    homo = bev.ground_homography(np.asarray(frame.intrinsic, float), cam_from_ego, 0.0)
    return {
        "cam_from_ego": np.round(cam_from_ego, 6).tolist(), "estimated": estimated,
        "homography": (homo / np.linalg.norm(homo)).round(10).tolist(), "x_range": list(bev.BEV2D_X),
        "y_range": list(bev.BEV2D_Y), "res": 0.1,
        "camera_height": round(float(np.linalg.inv(cam_from_ego)[2, 3]), 3),
    }  # fmt: skip


@router.get("/frames/{frame_id}/bev")
def get_bev(frame_id: str, request: Request, store: WorkspaceStore = Depends(get_store),
            dataroot: Path = Depends(get_dataroot)):  # fmt: skip
    """Ảnh camera của frame chiếu xuống mặt đường (homography), PNG RGBA; phía trước ở trên."""
    from src.services import bev

    frame = _load(store, frame_id)
    cache = store.root / "bev2d" / f"{store.check_id(frame_id)}.png"
    if not cache.exists():
        img = bev.load_rgb(image_path(dataroot, frame, 0, workspace=store.root))
        if img is None:
            raise _error(404, "IMAGE_NOT_FOUND", "Không tìm thấy file ảnh trong dataroot")
        cam_from_ego, _ = _cam_from_ego(frame, request)
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
    before = frame.model_dump_json()
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        entry = review.apply_action(frame, req, reviewer, config)
    except review.ReviewError as e:
        raise _error(422, "INVALID_ACTION", str(e)) from e
    store.save_frame(frame)
    store.append_corrections([entry])
    history.record(store, history.kind_2d(), frame_id, before)
    return frame


@router.post("/frames/{frame_id}/sweeps/{offset}/actions", response_model=FrameRecord)
def sweep_action(
    frame_id: str,
    offset: int,
    req: SweepActionRequest,
    store: WorkspaceStore = Depends(get_store),
    dataroot: Path = Depends(get_dataroot),
    config: AutoLabelConfig = Depends(get_config),
):
    """Sửa một box ở sweep t±n (giữ / xoá / đổi lớp / sửa / vẽ thêm) rồi tính lại QA và score của keyframe."""
    from src.services.pipeline import resolve_image
    from src.services.propagation import lidar_arrays
    from src.services.sweep_review import apply_sweep_action, requalify_frame

    frame = _load(store, frame_id)
    before = frame.model_dump_json()
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        entry = apply_sweep_action(frame, offset, req, reviewer, config)
    except review.ReviewError as e:
        raise _error(422, "INVALID_ACTION", str(e)) from e
    uv, depth = lidar_arrays(store.load_aux("lidar", frame_id))
    requalify_frame(frame, config, lambda p: resolve_image(dataroot, store.root, p), uv, depth)
    store.save_frame(frame)
    store.append_event(dict(type="sweep_action", mode="2d", **entry))
    history.record(store, history.kind_2d(), frame_id, before)
    return frame


@router.post("/frames/{frame_id}/approve-low-risk", response_model=FrameRecord)
def approve_low_risk(frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    before = frame.model_dump_json()
    try:
        entries = review.approve_low_risk(frame, req.reviewer or get_settings().reviewer_name)
    except review.ReviewError as e:
        raise _error(409, "FRAME_APPROVED", str(e)) from e
    store.save_frame(frame)
    store.append_corrections(entries)
    if entries:
        history.record(store, history.kind_2d(), frame_id, before)
    return frame


@router.post("/frames/{frame_id}/approve", response_model=FrameRecord)
def approve_frame(frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        review.approve_frame(frame, reviewer, req.review_time_s)
    except review.ReviewError as e:
        raise _error(409, "PENDING_OBJECTS", str(e)) from e
    store.save_frame(frame)
    store.append_event(dict(type="approve", mode="2d", frame_id=frame_id, reviewer=reviewer,
                            review_time_s=req.review_time_s, n_objects=len(frame.objects)))  # fmt: skip
    return frame


@router.post("/frames/{frame_id}/reject", response_model=FrameRecord)
def reject_frame(frame_id: str, req: RejectRequest, store: WorkspaceStore = Depends(get_store)):
    """Reviewer trả lại frame, bắt buộc có lý do (FR-15)."""
    frame = _load(store, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        review.reject_frame(frame, reviewer, req.reason)
    except review.ReviewError as e:
        raise _error(422, "REASON_REQUIRED", str(e)) from e
    store.save_frame(frame)
    store.append_event(dict(type="reject", mode="2d", frame_id=frame_id, reviewer=reviewer, reason=frame.reject_reason))
    return frame


def undo_redo(store: WorkspaceStore, kind: str, frame, direction: str, reviewer: str, model_cls, save, log) -> object:
    """Undo / redo dùng chung cho 2D (FrameRecord) và 3D (Frame3DRecord)."""
    if frame.status == "approved":
        raise _error(409, "FRAME_APPROVED", "Frame đã approve, mở lại trước khi hoàn tác")
    try:
        target = history.step(store, kind, frame.frame_id, frame.model_dump_json(), direction)
    except history.HistoryError as e:
        raise _error(409, "NOTHING_TO_" + direction.upper(), str(e)) from e
    restored = model_cls.model_validate_json(target)
    if restored.status == "approved":  # bản cũ được lưu lúc đang sửa; không bao giờ tự approve lại
        restored.status = "editing"
    save(restored)
    log([o for o in history.changed_objects(frame, restored)], restored, direction.upper(), reviewer)
    store.append_event(dict(type=direction, mode="3d" if kind != history.kind_2d() else "2d",
                            frame_id=frame.frame_id, reviewer=reviewer))  # fmt: skip
    return restored


@router.post("/frames/{frame_id}/undo", response_model=FrameRecord)
@router.post("/frames/{frame_id}/redo", response_model=FrameRecord)
def undo_frame(frame_id: str, req: ReviewerRequest, request: Request, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name

    def log(objs, restored, action, who):
        store.append_corrections(
            [dict(review._log_entry(restored, o, action, who), timestamp=review.now_iso()) for o in objs]
        )

    direction = "redo" if request.url.path.endswith("/redo") else "undo"
    return undo_redo(store, history.kind_2d(), frame, direction, reviewer, FrameRecord, store.save_frame, log)


@router.get("/frames/{frame_id}/history")
def frame_history(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, frame_id)
    return history.counts(store, history.kind_2d(), frame_id)


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
    store.append_event(dict(type="reopen", mode="2d", frame_id=frame_id, reviewer=get_settings().reviewer_name))
    return frame


@router.get("/corrections")
def corrections(
    frame_id: str | None = None, limit: int = Query(200, ge=1, le=5000), store: WorkspaceStore = Depends(get_store)
):
    return store.corrections(frame_id)[-limit:][::-1]


@router.get("/metrics")
def metrics(store: WorkspaceStore = Depends(get_store)):
    from src.services import productivity

    frames = store.list_frames()
    m = review.compute_metrics(frames)
    m["productivity"] = {
        "reviewers": productivity.reviewer_stats(frames, store.events(), store.corrections(), "2d"),
        "inference": productivity.inference_stats(frames),
    }
    return m


@router.get("/report.csv")
def report_csv(kind: Literal["frames", "summary"] = "frames", store: WorkspaceStore = Depends(get_store)):
    """Báo cáo CSV (FR-19): kind=frames bảng từng frame, kind=summary số liệu tổng hợp (gồm năng suất)."""
    from fastapi.responses import Response

    from src.services import productivity

    body = (
        productivity.frames_csv(store.list_frames(), "2d")
        if kind == "frames"
        else productivity.summary_csv(metrics(store))
    )
    return Response(
        "\ufeff" + body,  # BOM: Excel mở đúng tiếng Việt
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="autolabel2d_{kind}.csv"'},
    )


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
