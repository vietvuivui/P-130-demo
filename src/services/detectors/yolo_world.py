"""YOLO-World (ultralytics): open-vocab, real-time."""

from __future__ import annotations

from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device


class YoloWorldDetector(Detector):
    name = "yolo_world"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        from ultralytics import YOLOWorld

        self.cfg = config.detection.yolo_world
        self.device = pick_device()
        self.model = YOLOWorld(self.cfg.weights)
        self.model.set_classes(self.prompts)

    @property
    def version(self) -> str:
        return f"{self.name}-{Path(self.cfg.weights).stem}-{self.cfg.imgsz}"

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        results = self.model.predict(
            [str(p) for p in image_paths],
            conf=self.config.detection.score_threshold,
            imgsz=self.cfg.imgsz,
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
                label = self.prompt_to_class[self.prompts[int(cls_idx)]]
                score = round(float(conf), 4)
                dets.append(
                    Detection(bbox=[round(v, 1) for v in xyxy], label=label, score=score, models={self.name: score})
                )
            out.append(dets)
        return out
