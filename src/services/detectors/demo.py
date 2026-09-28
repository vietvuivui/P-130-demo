"""Detector demo: tìm vật thể theo màu trên video tổng hợp của `python -m src.demo`. Không cần GPU/torch.

Chỉ dùng cho demo và test: video demo vẽ mỗi loại vật thể bằng một màu riêng, detector này tìm các vùng
cùng màu (connected components) rồi trả box như một detector thật, kể cả những "lỗi" có chủ ý để demo QA:
- xe màu cam được nhận là truck nhưng phân vân với car (CLASS_CONFLICT) -> người sửa lớp, lan truyền mang theo;
- vật nhỏ ở xa có score thấp (LOW_CONFIDENCE);
- vệt đỏ loé lên ở một frame (FLICKER).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector

# màu (RGB) -> (lớp, lớp phân vân + tỉ lệ score). Dùng chung với phần vẽ video trong src/demo.py
DEMO_PALETTE: dict[str, dict] = {
    "car_red": {"rgb": (214, 38, 38), "label": "car"},
    "car_blue": {"rgb": (36, 92, 226), "label": "car"},
    "van_orange": {"rgb": (242, 146, 24), "label": "truck", "alt": ("car", 0.85)},
    "pedestrian": {"rgb": (240, 214, 40), "label": "pedestrian"},
    "cyclist": {"rgb": (150, 60, 205), "label": "bicycle"},
}


class DemoColorDetector(Detector):
    name = "demo"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        self.cfg = config.detection.demo

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        import cv2

        out = []
        for path in image_paths:
            # imdecode(fromfile) thay cho imread: đọc được cả đường dẫn Unicode trên Windows
            data = np.fromfile(str(path), dtype=np.uint8) if Path(path).is_file() else None
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None and data.size else None
            if bgr is None:
                out.append([])
                continue
            rgb = bgr[:, :, ::-1].astype(np.int16)
            dets = []
            for spec in DEMO_PALETTE.values():
                mask = (np.abs(rgb - np.array(spec["rgb"], dtype=np.int16)).max(axis=2) <= self.cfg.tolerance).astype(
                    np.uint8
                )
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for x, y, w, h, area in stats[1:n]:
                    if area < self.cfg.min_area:
                        continue
                    # Vật càng nhỏ (càng xa) càng kém chắc chắn, giống detector thật
                    score = round(float(np.clip(0.3 + 0.65 * min(1.0, area / self.cfg.confident_area), 0.2, 0.95)), 4)
                    bbox = [float(x), float(y), float(x + w), float(y + h)]
                    dets.append(Detection(bbox=bbox, label=spec["label"], score=score, models={self.name: score}))
                    if "alt" in spec:
                        # Giống detector thật phân vân: cùng box, lớp khác, score thấp hơn. Bước fusion gộp hai box
                        # này thành một object có alternatives -> CLASS_CONFLICT
                        alt_label, ratio = spec["alt"]
                        alt_score = round(score * ratio, 4)
                        dets.append(
                            Detection(bbox=bbox, label=alt_label, score=alt_score, models={self.name: alt_score})
                        )
            out.append(dets)
        return out
