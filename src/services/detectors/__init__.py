"""Detector 2D open-vocab + ensemble có cache theo ảnh."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector
from src.services.detectors.fusion import fuse_detections


class _DropHalfDeprecation(logging.Filter):
    """Ultralytics 8.4 in cảnh báo "'half' is deprecated" ở MỖI lần predict (hàng nghìn dòng khi chạy cả scene).
    half=True vẫn chạy đúng trên GPU; chỉ bỏ dòng cảnh báo lặp lại."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "'half' is deprecated" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_DropHalfDeprecation())

# Thư viện mỗi detector cần (ngoài requirements.txt): báo lỗi dễ hiểu thay cho ModuleNotFoundError
DETECTOR_PACKAGES = {
    "yolo_world": ["ultralytics", "torch"], "yoloe": ["ultralytics", "torch"], "yolo26": ["ultralytics", "torch"],
    "grounding_dino": ["transformers", "torch"], "florence2": ["transformers", "torch"], "demo": [],
}  # fmt: skip


# Gói của requirements.txt mà các bước xử lý cần (máy cài từ bản cũ có thể thiếu): module -> tên gói pip
CORE_PACKAGES = {"scipy": "scipy", "cv2": "opencv-python-headless", "multipart": "python-multipart",
                 "langgraph": "langgraph"}  # fmt: skip


def missing_core() -> list[str]:
    import importlib.util

    return sorted(pip for mod, pip in CORE_PACKAGES.items() if importlib.util.find_spec(mod) is None)


def missing_packages(names: list[str]) -> list[str]:
    """Thư viện còn thiếu để chạy các detector `names` (kiểm tra nhanh, không import)."""
    import importlib.util

    need = {pkg for n in names for pkg in DETECTOR_PACKAGES.get(n, [])}
    return sorted(pkg for pkg in need if importlib.util.find_spec(pkg) is None)


class MissingPackagesError(RuntimeError):
    def __init__(self, missing: list[str]):
        import sys

        super().__init__(
            f"Python đang chạy server ({sys.executable}) thiếu thư viện {', '.join(missing)}. "
            "Cài bằng: python -m pip install -r requirements-ml.txt (gồm cả requirements.txt; torch bản GPU xem "
            "README), hoặc chạy server bằng Python đã cài sẵn: <python đó> -m uvicorn src.main:app"
        )
        self.missing = missing


def build_detector(name: str, config: AutoLabelConfig) -> Detector:
    missing = missing_packages([name])
    if missing:
        raise MissingPackagesError(missing)
    # Import lười để phần còn lại của app không cần torch
    if name == "yolo_world":
        from src.services.detectors.yolo_world import YoloWorldDetector

        return YoloWorldDetector(config)
    if name == "yoloe":
        from src.services.detectors.yoloe import YoloeDetector

        return YoloeDetector(config)
    if name == "yolo26":
        from src.services.detectors.yolo26 import Yolo26Detector

        return Yolo26Detector(config)
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


_SHARED: dict[tuple[str, str], Detector] = {}
_SHARED_LOCK = threading.Lock()


class DetectorEnsemble:
    """Chạy các detector trên một lô ảnh, cache kết quả thô theo (model, sample_data token)."""

    def __init__(self, config: AutoLabelConfig, cache_dir: Path, names: list[str] | None = None):
        self.config = config
        self.names = names or config.detection.detectors
        self.cache_dir = cache_dir
        self._detectors: dict[str, Detector] = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> Detector:
        # Model dùng chung giữa các ensemble có cùng cấu hình (mỗi dự án một workspace/cache, nhưng chỉ nạp model một
        # lần vào GPU)
        with _SHARED_LOCK:
            if name not in self._detectors:
                key = (name, self._cache_file(name, "_").parent.name)
                if key not in _SHARED:
                    _SHARED[key] = build_detector(name, self.config)
                self._detectors[name] = _SHARED[key]
            return self._detectors[name]

    def _cache_file(self, name: str, sd_token: str) -> Path:
        # Khoá theo tên model + cấu hình liên quan, đổi prompt/weights/ngưỡng thì cache tự vô hiệu
        det_cfg = getattr(self.config.detection, name)
        dump = det_cfg.model_dump()
        if not dump.get("tta_flip"):  # tuỳ chọn mới, tắt thì giữ khoá cũ để cache đã có vẫn dùng được
            dump.pop("tta_flip", None)
        # mask không đổi box: giữ khoá cũ để khỏi detect lại cả workspace. Ảnh detect trước bản có mask thì không có
        # mask; muốn có thì xoá thư mục cache/detections/yoloe-* rồi chạy lại
        dump.pop("masks", None)
        raw = json.dumps([dump, self.config.prompt_to_class(), self.config.detection.score_threshold])
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
        return fuse_detections(per_model, self.config.detection.fusion_iou)

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
