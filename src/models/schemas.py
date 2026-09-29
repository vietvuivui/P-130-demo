"""Schema nhãn 2D: detection, kết quả QA, trạng thái review, frame, API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
# qc: issue do bước QC gắn thêm sau khi người đã duyệt (vd. audit ngẫu nhiên phát hiện sai)
IssueGroup = Literal["detection", "lidar", "temporal", "geometric", "qc"]
HumanAction = Literal["KEEP", "DELETE", "CHANGE_CLASS", "EDIT_BOX", "ADD_BOX", "BATCH_APPROVE", "PROPAGATED_DELETE"]
ObjectSource = Literal["model", "track", "human", "propagated"]


class Detection(BaseModel):
    """Một box 2D do model (hoặc tracking) sinh ra, [x1, y1, x2, y2] theo pixel."""

    bbox: list[float] = Field(..., min_length=4, max_length=4)
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    # Score của lớp chính theo từng model đã thấy object
    models: dict[str, float] = Field(default_factory=dict)
    # Lớp khác mà model gán cho cùng vị trí -> nguồn của CLASS_CONFLICT
    alternatives: dict[str, float] = Field(default_factory=dict)


class QAIssue(BaseModel):
    code: str
    group: IssueGroup
    message: str


class QAResult(BaseModel):
    risk: float
    level: RiskLevel
    issues: list[QAIssue] = Field(default_factory=list)
    # Bốn thành phần của risk score, mỗi cái trong [0, 1]
    terms: dict[str, float] = Field(default_factory=dict)
    lidar: dict = Field(default_factory=dict)
    temporal: dict = Field(default_factory=dict)


class ReviewState(BaseModel):
    status: Literal["pending", "approved", "deleted"] = "pending"
    action: HumanAction | None = None
    final_label: str | None = None
    final_bbox: list[float] | None = None
    reviewer: str | None = None
    at: str | None = None


class PropagationInfo(BaseModel):
    """Object này được lan truyền từ quyết định của người ở một keyframe trước (FR-11 → FR-13)."""

    keyframe_id: str
    keyframe_object_id: str
    # Độ tin cậy lan truyền c_prop trong [0, 1]
    prop_conf: float
    # Detector có thấy object ở frame này không; False = box dự đoán từ chuyển động
    matched: bool = True
    # Số ảnh camera (keyframe + sweep) đã đi qua kể từ keyframe gốc
    steps: int = 0
    # Lớp và score detector gán ở frame này, nếu khác lớp lan truyền
    detector_label: str | None = None
    detector_score: float | None = None


class LabelObject(BaseModel):
    object_id: str
    bbox: list[float]
    label: str
    score: float
    # model: box từ detector; track: box nội suy (RECOVERED_BY_TRACK); human: box người vẽ;
    # propagated: lan truyền từ keyframe người đã duyệt
    source: ObjectSource = "model"
    # Định danh object xuyên suốt scene (giữ nguyên qua các frame nhờ lan truyền)
    track_id: str | None = None
    propagation: PropagationInfo | None = None
    models: dict[str, float] = Field(default_factory=dict)
    alternatives: dict[str, float] = Field(default_factory=dict)
    # Box của cùng object ở các sweep lân cận, key là offset ("-2", "-1", "1", "2")
    track: dict[str, list[float] | None] = Field(default_factory=dict)
    qa: QAResult | None = None
    review: ReviewState = Field(default_factory=ReviewState)


class ImageInfo(BaseModel):
    sd_token: str
    path: str  # tương đối so với dataroot
    timestamp: int
    width: int = 1600
    height: int = 900


class SweepInfo(ImageInfo):
    offset: int
    detections: list[Detection] = Field(default_factory=list)


QCSeverity = Literal["error", "warning"]


class QCFinding(BaseModel):
    """Một lỗi QC trên nhãn cuối (sau khi người sửa) hoặc trên file nhãn kiểm nhanh.

    error: nhãn hỏng, phải sửa. warning: đáng ngờ, sửa hoặc xác nhận "đã kiểm, giữ nguyên" (ack).
    """

    code: str
    severity: QCSeverity
    message: str
    frame_id: str
    object_id: str | None = None
    # Object thứ hai của cặp (box trùng, chồng khác lớp)
    other_object_id: str | None = None
    # Đề xuất sửa: {"action": "DELETE" | "EDIT_BOX" | "ADD_BOX" | "CHANGE_CLASS", "object_id"?, "bbox"?, "label"?}
    suggestion: dict | None = None
    # key: định danh finding trong frame; fingerprint: băm trạng thái cuối của object liên quan.
    # Ack chỉ còn hiệu lực khi cả hai khớp: sửa box sau khi ack thì phải kiểm lại
    key: str
    fingerprint: str
    acked: bool = False


class QCAck(BaseModel):
    key: str
    fingerprint: str
    reviewer: str
    at: str
    note: str = ""


class FrameRecord(BaseModel):
    frame_id: str
    sample_token: str
    scene: str
    index: int
    camera: str
    image: ImageInfo
    intrinsic: list[list[float]]
    sweeps: list[SweepInfo] = Field(default_factory=list)
    detectors: list[str] = Field(default_factory=list)
    has_lidar: bool = True
    status: Literal["auto", "editing", "approved"] = "auto"
    frame_risk: float = 0.0
    objects: list[LabelObject] = Field(default_factory=list)
    created_at: str | None = None
    opened_at: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None
    review_time_s: float | None = None
    # Lan truyền: frame gốc và lúc lan truyền. prelabel giữ bản pre-label trước lần lan truyền
    # đầu tiên, để lan truyền lại (từ keyframe khác) luôn bắt đầu từ cùng một điểm.
    propagated_from: str | None = None
    propagated_at: str | None = None
    prelabel: list[LabelObject] | None = None
    # QC: xác nhận của người cho warning QC, và finding do audit ngẫu nhiên gắn vào (vd. frame còn vật bị sót)
    qc_acks: list[QCAck] = Field(default_factory=list)
    qc_manual: list[QCFinding] = Field(default_factory=list)


class FrameSummary(BaseModel):
    frame_id: str
    scene: str
    index: int
    status: str
    frame_risk: float
    counts: dict[str, int]
    pending: int
    n_objects: int
    propagated_from: str | None = None


# ---- API ----


class ReviewActionRequest(BaseModel):
    action: Literal["KEEP", "DELETE", "CHANGE_CLASS", "EDIT_BOX", "ADD_BOX"]
    object_id: str | None = None
    label: str | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reviewer: str | None = None


class ReviewerRequest(BaseModel):
    reviewer: str | None = None
    review_time_s: float | None = Field(default=None, ge=0)


class PropagateRequest(BaseModel):
    # Số keyframe tối đa đi tới; None = theo config
    max_frames: int | None = Field(default=None, ge=1, le=200)


class PropagateSkip(BaseModel):
    frame_id: str
    reason: str


class PropagateResponse(BaseModel):
    keyframe_id: str
    frames_updated: list[str] = Field(default_factory=list)
    frames_skipped: list[PropagateSkip] = Field(default_factory=list)
    # Frame người đã mở/duyệt mà lan truyền dừng lại trước nó
    stopped_at: str | None = None
    stop_reason: str | None = None
    tracks_started: int = 0
    tracks_alive: int = 0
    objects_propagated: int = 0
    objects_suppressed: int = 0
    # Ảnh camera chưa có detection trong cache (tracker chỉ dự đoán ở các ảnh này)
    images_without_detections: int = 0


# ---- Video ----
# Một video = chuỗi ảnh theo thời gian. Keyframe (có FrameRecord) là frame để gán nhãn; ảnh giữa hai keyframe
# chỉ dùng để tracking khi lan truyền. Scene nuScenes: keyframe 2Hz + sweep 12Hz. Video mp4 tải lên: cắt frame ở
# track_fps, cứ (track_fps / label_fps) frame thì có một keyframe.


class TimelineEntry(BaseModel):
    sd_token: str
    timestamp: int  # micro giây
    sample_token: str | None = None  # chỉ keyframe
    path: str = ""


class VideoRecord(BaseModel):
    """Video tải lên (mp4). Scene nuScenes không cần record: timeline đọc thẳng từ bảng nuScenes."""

    video_id: str
    name: str
    source: Literal["upload"] = "upload"
    status: Literal["processing", "ready", "error"] = "processing"
    progress: float = 0.0
    message: str | None = None
    width: int = 0
    height: int = 0
    track_fps: float = 10.0
    label_fps: float = 2.0
    duration_s: float = 0.0
    detectors: list[str] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    created_at: str | None = None


class VideoFrame(FrameSummary):
    timestamp: int
    t: float  # giây kể từ đầu video


class VideoSummary(BaseModel):
    video_id: str
    name: str
    source: Literal["nuscenes", "upload"]
    status: Literal["processing", "ready", "error"] = "ready"
    progress: float = 1.0
    message: str | None = None
    n_frames: int = 0
    approved: int = 0
    editing: int = 0
    propagated: int = 0
    pending_objects: int = 0
    duration_s: float = 0.0
    first_frame_id: str | None = None


class VideoDetail(VideoSummary):
    frames: list[VideoFrame] = Field(default_factory=list)


class ExportResponse(BaseModel):
    export_id: str
    n_frames: int
    n_objects: int
    files: list[str]
    # READY / NOT_READY theo checklist QC lúc xuất (chi tiết trong manifest.json và qa_report.md)
    release_status: str | None = None
    # Frame đã approve nhưng nhãn cuối còn lỗi QC chưa xử lý -> không xuất
    frames_skipped_qc: list[str] = Field(default_factory=list)


# ---- QC ----


class FrameQCResponse(BaseModel):
    frame_id: str
    findings: list[QCFinding] = Field(default_factory=list)
    # Số finding chưa sửa / chưa ack (chặn approve khi > 0)
    open: int = 0


class QCAckRequest(BaseModel):
    key: str
    fingerprint: str
    note: str = Field(default="", max_length=300)
    reviewer: str | None = None


class AuditItem(BaseModel):
    """Một mẫu audit: object đã duyệt theo lô (kind=object) hoặc frame đã approve (kind=frame, kiểm vật bị sót)."""

    audit_id: str
    kind: Literal["object", "frame"]
    frame_id: str
    object_id: str | None = None
    seed: int
    created_at: str
    # Nhãn lúc lấy mẫu (object) để người audit thấy đúng thứ đã được duyệt
    label: str | None = None
    bbox: list[float] | None = None
    result: Literal["ok", "error"] | None = None
    note: str = ""
    reviewer: str | None = None
    at: str | None = None


class AuditSampleRequest(BaseModel):
    kind: Literal["object", "frame"] = "object"
    # None = theo config (object_sample_size / frame_sample_size)
    size: int | None = Field(default=None, ge=1, le=1000)
    seed: int | None = None


class AuditResultRequest(BaseModel):
    result: Literal["ok", "error"]
    note: str = Field(default="", max_length=300)
    reviewer: str | None = None


class QuickCheckLabel(BaseModel):
    object_id: str
    label: str
    bbox: list[float]


class QuickCheckFrame(BaseModel):
    frame_id: str | None = None
    file_name: str | None = None
    in_workspace: bool = False
    width: int | None = None
    height: int | None = None
    labels: list[QuickCheckLabel] = Field(default_factory=list)
    findings: list[QCFinding] = Field(default_factory=list)


class QuickCheckResponse(BaseModel):
    file_name: str
    format: str
    n_frames: int
    n_labels: int
    n_errors: int
    n_warnings: int
    # Cảnh báo file ghi là người đã xem và giữ nguyên (qc_acknowledged, issue của nhãn KEEP): không tính là lỗi mở
    n_acked: int = 0
    # Chỉ đếm finding còn mở
    by_code: dict[str, int] = Field(default_factory=dict)
    frames: list[QuickCheckFrame] = Field(default_factory=list)
    elapsed_ms: float
    # Lỗi đọc file không gắn được vào frame/nhãn nào (dòng JSON hỏng, ...)
    parse_errors: list[str] = Field(default_factory=list)
