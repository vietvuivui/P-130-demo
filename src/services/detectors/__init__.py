"""Detector 2D open-vocab + ensemble có cache theo ảnh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector
from src.services.detectors.fusion import fuse_detections


def build_detector(name: str, config: AutoLabelConfig) -> Detector:
    # Import lười để phần còn lại của app không cần torch
    if name == "yolo_world":
        from src.services.detectors.yolo_world import YoloWorldDetector

        return YoloWorldDetector(config)
    if name == "grounding_dino":
        from src.services.detectors.grounding_dino import GroundingDinoDetector

        return GroundingDinoDetector(config)
    if name == "florence2":
        from src.services.detectors.florence2 import Florence2Detector

        return Florence2Detector(config)
    raise ValueError(f"Detector không hỗ trợ: {name}")


class DetectorEnsemble:
    """Chạy các detector trên một lô ảnh, cache kết quả thô theo (model, sample_data token)."""

    def __init__(self, config: AutoLabelConfig, cache_dir: Path, names: list[str] | None = None):
        self.config = config
        self.names = names or config.detection.detectors
        self.cache_dir = cache_dir
        self._detectors: dict[str, Detector] = {}

    def _get(self, name: str) -> Detector:
        if name not in self._detectors:
            self._detectors[name] = build_detector(name, self.config)
        return self._detectors[name]

    def _cache_file(self, name: str, sd_token: str) -> Path:
        # Khoá theo tên model + cấu hình liên quan, đổi prompt/weights/ngưỡng thì cache tự vô hiệu
        det_cfg = getattr(self.config.detection, name)
        raw = json.dumps([det_cfg.model_dump(), self.config.prompt_to_class(), self.config.detection.score_threshold])
        key = f"{name}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"
        return self.cache_dir / key / f"{sd_token}.json"

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
        return [fuse_detections(dets, self.config.detection.fusion_iou) for dets in per_image]
