"""YOLO26 thường (ultralytics): tập lớp đóng COCO 80.

Chỉ giữ các lớp COCO trùng tên một prompt trong taxonomy (car, truck, bus, person, bicycle, motorcycle); các lớp
barrier, traffic_cone, trailer, construction_vehicle không có trong COCO nên model này không bao giờ sinh ra.
Hợp làm model phụ nhanh trong ensemble, không hợp làm detector chính.
"""

from __future__ import annotations

from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device


class Yolo26Detector(Detector):
    name = "yolo26"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        from ultralytics import YOLO

        self.cfg = config.detection.yolo26
        self.device = pick_device()
        self.model = YOLO(self.cfg.weights)
        # chỉ số lớp COCO -> lớp nội bộ
        self.coco_to_class = {
            i: self.prompt_to_class[n] for i, n in self.model.names.items() if n in self.prompt_to_class
        }

    @property
    def version(self) -> str:
        return f"{self.name}-{Path(self.cfg.weights).stem}-{self.cfg.imgsz}"

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        results = self.model.predict(
            [str(p) for p in image_paths],
            conf=self.config.detection.score_threshold,
            imgsz=self.cfg.imgsz,
            classes=sorted(self.coco_to_class),
            device=self.device,
            half=self.device == "cuda",
            verbose=False,
        )
        out = []
        for r in results:
            dets = []
            for xyxy, conf, cls_idx in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True
            ):
                score = round(float(conf), 4)
                dets.append(
                    Detection(
                        bbox=[round(v, 1) for v in xyxy],
                        label=self.coco_to_class[int(cls_idx)],
                        score=score,
                        models={self.name: score},
                    )
                )
            out.append(dets)
        return out
