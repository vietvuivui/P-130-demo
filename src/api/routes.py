from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from src.config import get_settings
from src.models.qa_config import AutoLabelConfig, get_autolabel_config
from src.models.schemas import ExportResponse, FrameRecord, FrameSummary, ReviewActionRequest, ReviewerRequest
from src.services import review
from src.services.exporter import EXPORT_FILES, NothingToExportError, export_dataset
from src.services.pipeline import image_path
from src.services.store import WorkspaceStore

router = APIRouter()

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
        path = image_path(dataroot, frame, offset)
    except KeyError as e:
        raise _error(404, "SWEEP_NOT_FOUND", f"Frame không có sweep offset {offset}") from e
    if not path.exists():
        raise _error(404, "IMAGE_NOT_FOUND", "Không tìm thấy file ảnh trong dataroot")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.get("/frames/{frame_id}/lidar")
def get_lidar(frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, frame_id)
    return store.load_aux("lidar", frame_id) or {"u": [], "v": [], "d": []}


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
