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


class GroundingDinoCfg(BaseModel):
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.2
    text_threshold: float = 0.2


class Florence2Cfg(BaseModel):
    model_id: str = "microsoft/Florence-2-base"
    default_score: float = 0.5


class DetectionCfg(BaseModel):
    detectors: list[str] = ["yolo_world"]
    score_threshold: float = 0.1
    fusion_iou: float = 0.55
    yolo_world: YoloWorldCfg = YoloWorldCfg()
    grounding_dino: GroundingDinoCfg = GroundingDinoCfg()
    florence2: Florence2Cfg = Florence2Cfg()


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


class AutoLabelConfig(BaseModel):
    camera: str = "CAM_FRONT"
    sweep_offsets: list[int] = [-2, -1, 1, 2]
    detection: DetectionCfg = DetectionCfg()
    classes: dict[str, ClassSpec] = Field(default_factory=dict)
    gt_category_map: dict[str, str] = Field(default_factory=dict)
    qa: QACfg = QACfg()

    def prompt_to_class(self) -> dict[str, str]:
        """Map mỗi prompt văn bản về lớp nội bộ."""
        return {p.lower(): name for name, spec in self.classes.items() for p in spec.prompts}


def load_autolabel_config(path: str | Path) -> AutoLabelConfig:
    with open(path, encoding="utf-8") as f:
        return AutoLabelConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_autolabel_config(path: str) -> AutoLabelConfig:
    return load_autolabel_config(path)
