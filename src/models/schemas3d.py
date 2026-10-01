"""Schema nhãn 3D: box 3D trong hệ LiDAR của keyframe, kết luận kiểm chứng bằng camera, trạng thái duyệt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.models.schemas import ReviewState, RiskLevel

# Kết luận kiểm chứng một box 3D bằng camera (tên giữ như bộ kiểm chứng của nhóm 3D, scripts/verify_objects.py)
Verdict = Literal[
    "DUNG", "DUNG VAT, BOX LECH", "SAI LOP", "NGHI BAO NHAM", "CAMERA KHONG XAC NHAN", "CHUA DU THONG TIN"
]
Action3D = Literal["KEEP", "DELETE", "CHANGE_CLASS", "EDIT_BOX", "ADD_BOX", "BATCH_APPROVE"]


class Box3D(BaseModel):
    """Box 3D trong hệ toạ độ cảm biến LIDAR_TOP của keyframe (nuScenes: x phải, y trước, z lên, m); size = [w, l, h]."""

    center: list[float] = Field(..., min_length=3, max_length=3)
    size: list[float] = Field(..., min_length=3, max_length=3)
    yaw: float  # rad, quanh trục z, 0 = hướng x
    velocity: list[float] = Field(default_factory=lambda: [0.0, 0.0])


class Verify3D(BaseModel):
    """Kết quả kiểm chứng một box 3D bằng camera + LiDAR (không dùng nhãn gốc)."""

    verdict: Verdict
    level: RiskLevel
    comment: str
    camera: str | None = None  # camera nhiều thông tin nhất
    cameras: dict[str, float] = Field(default_factory=dict)  # điểm thông tin của từng camera thấy box
    bbox2d: dict[str, list[float]] = Field(default_factory=dict)  # box 3D chiếu xuống từng camera
    visible: float | None = None
    occlusion: float | None = None
    clarity: float | None = None
    # Theo từng camera thấy box: phần box nằm trong ảnh (0..1) và mức bị che (0..1); UI vẽ nét đứt khi bị che / cắt
    visible_by_cam: dict[str, float] = Field(default_factory=dict)
    occlusion_by_cam: dict[str, float] = Field(default_factory=dict)
    lidar_points: int = 0
    distance_m: float = 0.0
    det_label: str | None = None  # lớp detector 2D gán ở vị trí đó
    det_prompt: str | None = None
    det_score: float | None = None
    det_bbox: list[float] | None = None
    det_camera: str | None = None
    iou: float | None = None


class Propagation3DInfo(BaseModel):
    """Box này được lan truyền từ quyết định của người ở một keyframe 3D trước (src/services/propagation3d.py)."""

    keyframe_id: str
    keyframe_object_id: str
    distance_m: float  # lệch giữa vị trí dự đoán theo chuyển động và box mô hình ở frame này
    hops: int = 1


class Object3D(BaseModel):
    object_id: str
    label: str
    score: float
    box: Box3D
    source: Literal["model", "human", "propagated"] = "model"
    track_id: str | None = (
        None  # cùng vật qua các keyframe (tinh chỉnh theo track), dùng làm instance khi xuất nuScenes
    )
    original_box: Box3D | None = None  # box mô hình sinh ra, giữ lại khi người sửa box (EDIT_BOX)
    verify: Verify3D | None = None
    review: ReviewState = Field(default_factory=ReviewState)
    propagation: Propagation3DInfo | None = None


class Camera3D(BaseModel):
    sd_token: str
    path: str
    width: int = 1600
    height: int = 900
    intrinsic: list[list[float]]
    cam_from_lidar: list[list[float]]  # 4x4, đã bù chuyển động ego giữa hai timestamp


class Frame3DRecord(BaseModel):
    frame_id: str
    model: str  # mô hình 3D sinh pre-label
    sample_token: str
    scene: str
    index: int
    lidar_sd_token: str
    global_from_lidar: list[list[float]]
    cameras: dict[str, Camera3D] = Field(default_factory=dict)
    status: Literal["auto", "editing", "approved", "rejected"] = "auto"
    frame_risk: float = 0.0
    objects: list[Object3D] = Field(default_factory=list)
    created_at: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None
    review_time_s: float | None = None
    reject_reason: str | None = None
    rejected_by: str | None = None
    rejected_at: str | None = None
    autolabel_s: float | None = None
    autolabel_run: str | None = None
    # Giây theo từng bước (detect, lidar, flow, qa / detect_6cam, points, verify), src/services/timing.py
    autolabel_timing: dict[str, float] | None = None
    # Lan truyền: keyframe gốc, lúc lan truyền; prelabel = box mô hình trước lần lan truyền đầu (lan truyền lại từ đó)
    propagated_from: str | None = None
    propagated_at: str | None = None
    prelabel: list[Object3D] | None = None


class Frame3DSummary(BaseModel):
    frame_id: str
    scene: str
    index: int
    status: str
    frame_risk: float
    counts: dict[str, int]
    pending: int
    n_objects: int


class Action3DRequest(BaseModel):
    """KEEP / DELETE / CHANGE_CLASS (label) / EDIT_BOX (box, label tuỳ chọn) / ADD_BOX (box + label, không object_id)."""

    action: Literal["KEEP", "DELETE", "CHANGE_CLASS", "EDIT_BOX", "ADD_BOX"]
    object_id: str | None = None
    label: str | None = None
    box: Box3D | None = None
    reviewer: str | None = None
