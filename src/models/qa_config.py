"""Schema cho configs/autolabel.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ClassSpec(BaseModel):
    prompts: list[str]
    height_m: tuple[float, float]
    aspect_wh: tuple[float, float]
    color: str = "#64748b"


class YoloWorldCfg(BaseModel):
    weights: str = "yolov8l-worldv2.pt"
    imgsz: int = 1280


class YoloeCfg(BaseModel):
    # YOLOE-26: open-vocab trên nền YOLO26, text encoder MobileCLIP2 (tự tải từ GitHub của Ultralytics)
    weights: str = "yoloe-26l-seg.pt"
    imgsz: int = 1280


class Yolo26Cfg(BaseModel):
    # YOLO26 thường: tập lớp đóng COCO 80, chỉ giữ lớp trùng tên prompt (car, truck, bus, person, bicycle, motorcycle)
    weights: str = "yolo26l.pt"
    imgsz: int = 1280


class GroundingDinoCfg(BaseModel):
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.2
    text_threshold: float = 0.2


class Florence2Cfg(BaseModel):
    model_id: str = "microsoft/Florence-2-base"
    default_score: float = 0.5


class DemoDetectorCfg(BaseModel):
    # Detector theo màu cho video demo (python -m src.demo), không cần GPU
    tolerance: int = 48
    min_area: int = 60
    confident_area: int = 3500


class DetectionCfg(BaseModel):
    detectors: list[str] = ["yoloe"]
    score_threshold: float = 0.1
    # Lọc sau khi gộp box, trước QA Agent (không đổi khoá cache: đổi ngưỡng không phải detect lại)
    min_score: float = 0.1
    min_score_per_class: dict[str, float] = Field(default_factory=dict)
    fusion_iou: float = 0.55
    yolo_world: YoloWorldCfg = YoloWorldCfg()
    yoloe: YoloeCfg = YoloeCfg()
    yolo26: Yolo26Cfg = Yolo26Cfg()
    grounding_dino: GroundingDinoCfg = GroundingDinoCfg()
    florence2: Florence2Cfg = Florence2Cfg()
    demo: DemoDetectorCfg = DemoDetectorCfg()

    def keep(self, label: str, score: float) -> bool:
        return score >= self.min_score_per_class.get(label, self.min_score)


class ConfidenceCfg(BaseModel):
    low_threshold: float = 0.35
    class_conflict_ratio: float = 0.6


class LidarCfg(BaseModel):
    min_points: int = 3
    min_box_height_px: float = 60
    min_depth_m: float = 1.0
    max_depth_m: float = 45.0
    size_tolerance: tuple[float, float] = (0.5, 1.5)


class TemporalCfg(BaseModel):
    match_iou: float = 0.3
    min_support: int = 2
    recover_min_score: float = 0.35


class GeometryCfg(BaseModel):
    max_area_ratio: float = 0.35
    border_margin_px: float = 4


class RiskWeights(BaseModel):
    detection: float = 0.35
    lidar: float = 0.30
    temporal: float = 0.20
    geometric: float = 0.15


class RiskLevels(BaseModel):
    medium: float = 0.30
    high: float = 0.60


class RiskCfg(BaseModel):
    weights: RiskWeights = RiskWeights()
    class_conflict_penalty: float = 0.30
    issue_floor: float = 0.30
    levels: RiskLevels = RiskLevels()


class QACfg(BaseModel):
    confidence: ConfidenceCfg = ConfidenceCfg()
    lidar: LidarCfg = LidarCfg()
    temporal: TemporalCfg = TemporalCfg()
    geometry: GeometryCfg = GeometryCfg()
    risk: RiskCfg = RiskCfg()


class PropagationWeights(BaseModel):
    agreement: float = 0.5
    continuity: float = 0.3
    lidar: float = 0.2


class PropagationCfg(BaseModel):
    match_iou: float = 0.3
    suppress_iou: float = 0.5
    smoothing: float = 0.7
    velocity_smoothing: float = 0.5
    max_coast_images: int = 6
    min_visible: float = 0.3
    min_box_px: float = 6
    weights: PropagationWeights = PropagationWeights()
    hop_decay: float = 0.99
    flag_below: float = 0.6
    stop_below: float = 0.15
    # Track không khớp detection ở chính keyframe đích (đang "trôi" theo vận tốc) có được ghi ra không
    emit_coasting: bool = False
    max_keyframes: int = 20
    class_differs_score: float = 0.5


class VideoCfg(BaseModel):
    # Video mp4 tải lên: cắt frame ở track_fps để tracking, cứ track_fps/label_fps frame có một keyframe để gán nhãn
    track_fps: float = 10.0
    label_fps: float = 2.0
    max_seconds: float = 120.0
    max_upload_mb: int = 500
    # Check temporal (FLICKER, RECOVERED_BY_TRACK) chỉ dùng frame lân cận cách keyframe không quá ngần này (giây).
    # Ngưỡng QA temporal được chỉnh cho sweep 12Hz (±167 ms); video thưa (vd. chỉ có keyframe 2 Hz) thì bỏ check
    # thay vì báo FLICKER sai cho mọi vật đang chuyển động
    max_sweep_gap_s: float = 0.25


class Verify3DCfg(BaseModel):
    # Box 3D dưới ngưỡng điểm này không đưa vào duyệt (như ngưỡng 0.3 nhóm 3D dùng)
    min_score: float = 0.3
    # Ngưỡng điểm của box 2D dùng để kiểm chứng
    det_conf: float = 0.2
    # Prompt "đối thủ": vật dễ nhầm với các lớp (lan can cố định, cột, biển báo...). Box của chúng bị bỏ
    distractors: list[str] = Field(
        default_factory=lambda: [
            "fence", "guardrail", "pole", "traffic sign", "fire hydrant", "trash can", "bollard",
            "mailbox", "stroller", "wheelchair", "kick scooter",
        ]
    )  # fmt: skip
    # Số lần quét LiDAR liền trước gộp thêm (nếu có trên đĩa) khi đếm điểm trong box / tính mức che
    lidar_sweeps: int = 4
    # Số điểm tối đa gửi lên UI mỗi keyframe
    max_points_ui: int = 60000


class AutoLabelConfig(BaseModel):
    camera: str = "CAM_FRONT"
    sweep_offsets: list[int] = [-2, -1, 1, 2]
    detection: DetectionCfg = DetectionCfg()
    classes: dict[str, ClassSpec] = Field(default_factory=dict)
    gt_category_map: dict[str, str] = Field(default_factory=dict)
    qa: QACfg = QACfg()
    propagation: PropagationCfg = PropagationCfg()
    video: VideoCfg = VideoCfg()
    verify3d: Verify3DCfg = Verify3DCfg()

    def prompt_to_class(self) -> dict[str, str]:
        """Map mỗi prompt văn bản về lớp nội bộ."""
        return {p.lower(): name for name, spec in self.classes.items() for p in spec.prompts}


def load_autolabel_config(path: str | Path) -> AutoLabelConfig:
    with open(path, encoding="utf-8") as f:
        return AutoLabelConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_autolabel_config(path: str) -> AutoLabelConfig:
    return load_autolabel_config(path)
