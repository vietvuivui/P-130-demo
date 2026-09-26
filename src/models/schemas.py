"""Schema nhãn 2D: detection, kết quả QA, trạng thái review, frame, API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
IssueGroup = Literal["detection", "lidar", "temporal", "geometric"]
HumanAction = Literal["KEEP", "DELETE", "CHANGE_CLASS", "EDIT_BOX", "ADD_BOX", "BATCH_APPROVE"]


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


class LabelObject(BaseModel):
    object_id: str
    bbox: list[float]
    label: str
    score: float
    # model: box từ detector; track: box nội suy (RECOVERED_BY_TRACK); human: box người vẽ
    source: Literal["model", "track", "human"] = "model"
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


class FrameSummary(BaseModel):
    frame_id: str
    scene: str
    index: int
    status: str
    frame_risk: float
    counts: dict[str, int]
    pending: int
    n_objects: int


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


class ExportResponse(BaseModel):
    export_id: str
    n_frames: int
    n_objects: int
    files: list[str]
