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


class Yolo26Cfg(BaseModel):
    weights: str = "weights/yolo26l.pt"
    imgsz: int = 1280


class YoloE26Cfg(BaseModel):
    weights: str = "weights/yoloe-26l-seg.pt"
    imgsz: int = 1280
    # Text encoder mã hoá prompt (MobileCLIP2-B, TorchScript) mà YOLOE-26 được train cùng
    text_encoder: str = "weights/mobileclip2_b.ts"


class YoloWorldCfg(BaseModel):
    weights: str = "yolov8l-worldv2.pt"
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
    detectors: list[str] = ["yoloe26"]
    score_threshold: float = 0.1
    fusion_iou: float = 0.55
    yolo26: Yolo26Cfg = Yolo26Cfg()
    yoloe26: YoloE26Cfg = YoloE26Cfg()
    yolo_world: YoloWorldCfg = YoloWorldCfg()
    grounding_dino: GroundingDinoCfg = GroundingDinoCfg()
    florence2: Florence2Cfg = Florence2Cfg()
    demo: DemoDetectorCfg = DemoDetectorCfg()


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


class QuickCheckCfg(BaseModel):
    max_file_mb: int = 20
    missing_min_score: float = 0.5
    missing_match_iou: float = 0.3
    disagree_min_score: float = 0.6


class AuditCfg(BaseModel):
    # Mẫu object duyệt theo lô (lỗi nhãn) và mẫu frame (vật bị sót). sample_size = 0: không bắt buộc audit loại đó
    object_sample_size: int = Field(default=50, ge=0)
    object_max_error_upper: float = 0.10
    frame_sample_size: int = Field(default=10, ge=0)
    frame_max_error_upper: float = 0.30
    confidence_z: float = 1.96


class QCCfg(BaseModel):
    gate_on_approve: bool = True
    min_box_px: float = 2
    duplicate_iou: float = 0.7
    cross_class_iou: float = 0.85
    cross_class_allowed: list[tuple[str, str]] = [("pedestrian", "bicycle"), ("pedestrian", "motorcycle")]
    quick_check: QuickCheckCfg = QuickCheckCfg()
    audit: AuditCfg = AuditCfg()


class AutoLabelConfig(BaseModel):
    camera: str = "CAM_FRONT"
    sweep_offsets: list[int] = [-2, -1, 1, 2]
    detection: DetectionCfg = DetectionCfg()
    classes: dict[str, ClassSpec] = Field(default_factory=dict)
    gt_category_map: dict[str, str] = Field(default_factory=dict)
    qa: QACfg = QACfg()
    qc: QCCfg = QCCfg()
    propagation: PropagationCfg = PropagationCfg()
    video: VideoCfg = VideoCfg()

    def prompt_to_class(self) -> dict[str, str]:
        """Map mỗi prompt văn bản về lớp nội bộ."""
        return {p.lower(): name for name, spec in self.classes.items() for p in spec.prompts}


def load_autolabel_config(path: str | Path) -> AutoLabelConfig:
    with open(path, encoding="utf-8") as f:
        return AutoLabelConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_autolabel_config(path: str) -> AutoLabelConfig:
    return load_autolabel_config(path)
