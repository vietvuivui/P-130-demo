import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.config import get_settings
from src.models.qa_config import AutoLabelConfig, get_autolabel_config
from src.models.schemas import (
    AuditResultRequest,
    AuditSampleRequest,
    ExportResponse,
    FrameQCResponse,
    FrameRecord,
    FrameSummary,
    PropagateRequest,
    PropagateResponse,
    QCAckRequest,
    QuickCheckResponse,
    ReviewActionRequest,
    ReviewerRequest,
    VideoDetail,
    VideoSummary,
)
from src.services import qc, review
from src.services import video as video_service
from src.services.detectors import DetectorEnsemble
from src.services.exporter import EXPORT_FILES, NothingToExportError, NotReadyError, export_dataset
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
    # QC nhãn cuối / Quick Check / audit
    "UNKNOWN_CLASS": "Lớp không có trong taxonomy (configs/autolabel.yaml): phải đổi sang lớp hợp lệ",
    "INVALID_BOX": "Box hỏng: toạ độ ngược, quá nhỏ hoặc nằm ngoài ảnh",
    "BOX_OUT_OF_IMAGE": "Box vượt ra ngoài ảnh: cắt về biên ảnh",
    "DUPLICATE_BOX": "Hai box cùng lớp gần như trùng nhau: một vật bị gán hai lần (hay gặp sau Add box / nhận box nội suy)",
    "OVERLAP_CROSS_CLASS": "Hai box khác lớp chồng khít lên nhau: một vật bị gán hai lớp",
    "POSSIBLY_MISSING": "Detector thấy vật ở đây, ổn định qua các sweep, nhưng file nhãn không có box nào",
    "MODEL_DISAGREES": "Detector gán lớp khác cho cùng vị trí với score cao",
    "FRAME_NOT_IN_WORKSPACE": "Frame không có trong workspace nên không có LiDAR / detection để đối chiếu",
    "AUDIT_FAILED": "Audit ngẫu nhiên phát hiện nhãn duyệt theo lô này bị sai: xem lại và sửa",
    "AUDIT_MISSING_OBJECT": "Audit ngẫu nhiên: frame còn vật chưa được gán nhãn. Vẽ thêm box rồi xác nhận",
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
        "qc": {
            "gate_on_approve": config.qc.gate_on_approve,
            "object_sample_size": config.qc.audit.object_sample_size,
            "frame_sample_size": config.qc.audit.frame_sample_size,
        },
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
def approve_frame(
    frame_id: str,
    req: ReviewerRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    frame = _load(store, frame_id)
    try:
        review.approve_frame(frame, req.reviewer or get_settings().reviewer_name, req.review_time_s)
    except review.ReviewError as e:
        raise _error(409, "PENDING_OBJECTS", str(e)) from e
    # QC nhãn cuối: nhãn người sửa / vẽ chưa đi qua kiểm tra nào -> không cho approve khi còn lỗi chưa xử lý
    if config.qc.gate_on_approve:
        pending_qc = qc.open_findings(qc.load_frame_findings(store, frame, config))
        if pending_qc:
            codes = ", ".join(sorted({f.code for f in pending_qc}))
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "QC_FINDINGS",
                    "message": f"Còn {len(pending_qc)} lỗi QC ở nhãn cuối ({codes}): sửa, hoặc xác nhận cảnh báo",
                    "findings": [f.model_dump() for f in pending_qc],
                },
            )
    store.save_frame(frame)
    return frame


# ---- QC ----


def _frame_qc(store: WorkspaceStore, frame: FrameRecord, config: AutoLabelConfig) -> FrameQCResponse:
    findings = qc.load_frame_findings(store, frame, config)
    return FrameQCResponse(frame_id=frame.frame_id, findings=findings, open=len(qc.open_findings(findings)))


@router.get("/frames/{frame_id}/qc", response_model=FrameQCResponse)
def frame_qc(frame_id: str, store: WorkspaceStore = Depends(get_store), config: AutoLabelConfig = Depends(get_config)):
    """QC nhãn cuối của frame: chạy lại sau mỗi thao tác của người (UI gọi sau khi sửa)."""
    return _frame_qc(store, _load(store, frame_id), config)


@router.post("/frames/{frame_id}/qc/ack", response_model=FrameQCResponse)
def ack_qc(
    frame_id: str,
    req: QCAckRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    """Xác nhận "đã kiểm, giữ nguyên" một cảnh báo QC. Lỗi (error) không xác nhận được, phải sửa."""
    frame = _load(store, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    at = review.now_iso()
    try:
        finding = qc.ack_finding(
            frame, qc.load_frame_findings(store, frame, config), req.key, req.fingerprint, reviewer, req.note, at
        )
    except qc.QCError as e:
        raise _error(409, "QC_ACK_REJECTED", str(e)) from e
    store.save_frame(frame)
    store.append_qc_events(
        [
            {
                "event": "QC_ACK",
                "frame_id": frame_id,
                "object_id": finding.object_id,
                "code": finding.code,
                "key": finding.key,
                "fingerprint": finding.fingerprint,
                "message": finding.message,
                "note": req.note,
                "reviewer": reviewer,
                "timestamp": at,
            }
        ]
    )
    return _frame_qc(store, frame, config)


@router.get("/qc/report")
def qc_report(store: WorkspaceStore = Depends(get_store), config: AutoLabelConfig = Depends(get_config)):
    """Checklist "sẵn sàng phát hành" của cả dataset + lỗi QC còn mở, track đổi lớp, lệch log, audit."""
    return qc.dataset_report(store, config)


@router.post("/qc/quick-check", response_model=QuickCheckResponse)
def quick_check(
    file: UploadFile = File(...),
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    """Kiểm nhanh file nhãn (COCO .json / .jsonl / {"frames": [...]}) — chỉ đọc, không ghi gì vào workspace."""
    name = file.filename or "labels.json"
    if Path(name).suffix.lower() not in (".json", ".jsonl"):
        raise _error(422, "QC_FORMAT", "Chỉ nhận file .json (COCO hoặc danh sách frame) hoặc .jsonl")
    limit = config.qc.quick_check.max_file_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise _error(413, "QC_FILE_TOO_LARGE", f"File lớn hơn {config.qc.quick_check.max_file_mb} MB")
    try:
        return qc.quick_check(store, config, name, data)
    except qc.QCError as e:
        raise _error(422, "QC_PARSE", str(e)) from e


def _audit_state(store: WorkspaceStore, config: AutoLabelConfig) -> dict:
    return {"items": store.load_audit(), "summary": qc.audit_summary(store, config)}


@router.get("/qc/audit")
def audit(store: WorkspaceStore = Depends(get_store), config: AutoLabelConfig = Depends(get_config)):
    return _audit_state(store, config)


@router.post("/qc/audit/sample")
def audit_sample(
    req: AuditSampleRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    """Lấy mẫu ngẫu nhiên (seed được lưu) object duyệt theo lô hoặc frame đã approve để người audit."""
    try:
        qc.create_sample(store, config, req.kind, req.size, req.seed)
    except qc.AuditError as e:
        raise _error(e.status, e.code, str(e)) from e
    return _audit_state(store, config)


@router.post("/qc/audit/{audit_id}")
def audit_result(
    audit_id: str,
    req: AuditResultRequest,
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    """Kết quả audit một mẫu. Sai -> frame mở lại, object quay về chờ duyệt (AUDIT_FAILED) / frame cần thêm box."""
    try:
        qc.record_result(store, config, audit_id, req.result, req.note, req.reviewer or get_settings().reviewer_name)
    except qc.AuditError as e:
        raise _error(e.status, e.code, str(e)) from e
    return _audit_state(store, config)


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
def export(
    require_ready: bool = Query(False, description="Chỉ xuất khi checklist QC là READY"),
    store: WorkspaceStore = Depends(get_store),
    config: AutoLabelConfig = Depends(get_config),
):
    try:
        return export_dataset(store, config, require_ready=require_ready)
    except NothingToExportError as e:
        raise _error(409, "NOTHING_TO_EXPORT", str(e)) from e
    except NotReadyError as e:
        raise _error(409, "NOT_READY", str(e)) from e


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
