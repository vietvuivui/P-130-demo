"""YOLOE-26 (ultralytics): open-vocab trên nền YOLO26, prompt bằng chữ như YOLO-World.

Text encoder là MobileCLIP2 (tải tự động từ GitHub của Ultralytics), không cần CLIP của OpenAI.
Model -seg trả cả mask; ở đây chỉ dùng box.
"""

from __future__ import annotations

from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device


class YoloeDetector(Detector):
    name = "yoloe"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        from ultralytics import YOLOE

        self.cfg = config.detection.yoloe
        self.device = pick_device()
        self.model = YOLOE(self.cfg.weights)
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

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
                score = round(float(conf), 4)
                dets.append(
                    Detection(
                        bbox=[round(v, 1) for v in xyxy],
                        label=self.prompt_to_class[self.prompts[int(cls_idx)]],
                        score=score,
                        models={self.name: score},
                    )
                )
            out.append(dets)
        return out
