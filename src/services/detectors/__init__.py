"""Detector 2D open-vocab + ensemble có cache theo ảnh."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector
from src.services.detectors.fusion import fuse_detections


def build_detector(name: str, config: AutoLabelConfig) -> Detector:
    # Import lười để phần còn lại của app không cần torch
    if name == "yolo26":
        from src.services.detectors.yolo26 import Yolo26Detector

        return Yolo26Detector(config)
    if name == "yoloe26":
        from src.services.detectors.yoloe26 import YoloE26Detector

        return YoloE26Detector(config)
    if name == "yolo_world":
        from src.services.detectors.yolo_world import YoloWorldDetector

        return YoloWorldDetector(config)
    if name == "grounding_dino":
        from src.services.detectors.grounding_dino import GroundingDinoDetector

        return GroundingDinoDetector(config)
    if name == "florence2":
        from src.services.detectors.florence2 import Florence2Detector

        return Florence2Detector(config)
    if name == "demo":
        from src.services.detectors.demo import DemoColorDetector

        return DemoColorDetector(config)
    raise ValueError(f"Detector không hỗ trợ: {name}")


def detector_classes(name: str, config: AutoLabelConfig) -> set[str] | None:
    """Lớp nội bộ mà detector có thể sinh ra; None = mọi lớp trong taxonomy (model open-vocab). Không cần torch."""
    if name == "yolo26":
        from src.services.detectors.yolo26 import coco_class_map

        return set(coco_class_map(config).values())
    return None


class DetectorEnsemble:
    """Chạy các detector trên một lô ảnh, cache kết quả thô theo (model, sample_data token)."""

    def __init__(self, config: AutoLabelConfig, cache_dir: Path, names: list[str] | None = None):
        self.config = config
        self.names = names or config.detection.detectors
        self.cache_dir = cache_dir
        # Lớp mỗi model phủ: fusion chỉ tính phiếu của model nhận được lớp đó
        self.coverage = {name: detector_classes(name, config) for name in self.names}
        self._detectors: dict[str, Detector] = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> Detector:
        with self._lock:  # hai request cùng lúc không nạp model hai lần
            if name not in self._detectors:
                self._detectors[name] = build_detector(name, self.config)
            return self._detectors[name]

    def _cache_file(self, name: str, sd_token: str) -> Path:
        # Khoá theo tên model + cấu hình liên quan, đổi prompt/weights/ngưỡng thì cache tự vô hiệu
        det_cfg = getattr(self.config.detection, name)
        raw = json.dumps([det_cfg.model_dump(), self.config.prompt_to_class(), self.config.detection.score_threshold])
        key = f"{name}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"
        return self.cache_dir / key / f"{sd_token}.json"

    def load_cached(self, sd_token: str) -> list[Detection] | None:
        """Detection đã fuse của một ảnh, chỉ đọc cache (không cần torch). None nếu có model chưa chạy ảnh này."""
        per_model: dict[str, list[Detection]] = {}
        for name in self.names:
            cache = self._cache_file(name, sd_token)
            if not cache.exists():
                return None
            per_model[name] = [Detection.model_validate(d) for d in json.loads(cache.read_text())]
        return fuse_detections(per_model, self.config.detection.fusion_iou, self.coverage)

    def detect_batch(self, images: list[tuple[str, Path]]) -> list[list[Detection]]:
        """images: list (sd_token, path). Trả về detection đã fuse cho từng ảnh."""
        per_image: list[dict[str, list[Detection]]] = [{} for _ in images]
        for name in self.names:
            missing = []
            for i, (tok, _) in enumerate(images):
                cache = self._cache_file(name, tok)
                if cache.exists():
                    per_image[i][name] = [Detection.model_validate(d) for d in json.loads(cache.read_text())]
                else:
                    missing.append(i)
            if not missing:
                continue
            results = self._get(name).detect([images[i][1] for i in missing])
            for i, dets in zip(missing, results, strict=True):
                per_image[i][name] = dets
                cache = self._cache_file(name, images[i][0])
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([d.model_dump() for d in dets]))
        return [fuse_detections(dets, self.config.detection.fusion_iou, self.coverage) for dets in per_image]
