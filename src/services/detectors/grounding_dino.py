"""Grounding DINO (HuggingFace transformers): open-vocab theo câu mô tả.

Bản 1.5 Edge chỉ có qua API; ở đây dùng bản open-weight grounding-dino-tiny/base.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device


class GroundingDinoDetector(Detector):
    name = "grounding_dino"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.cfg = config.detection.grounding_dino
        self.device = pick_device()
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.model_id).to(self.device).eval()
        self.text = " . ".join(self.prompts) + " ."
        # Tên tham số ngưỡng box đổi giữa các bản transformers (box_threshold -> threshold)
        params = inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self.threshold_kw = "threshold" if "threshold" in params else "box_threshold"

    @property
    def version(self) -> str:
        return f"{self.name}-{self.cfg.model_id.split('/')[-1]}"

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        out = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[image.size[::-1]],
                **{self.threshold_kw: max(self.cfg.box_threshold, self.config.detection.score_threshold)},
            )[0]
            phrases = res.get("text_labels", res.get("labels"))
            dets = []
            for box, score, phrase in zip(res["boxes"].tolist(), res["scores"].tolist(), phrases, strict=True):
                label = self.phrase_to_class(str(phrase))
                if label is None:
                    continue
                score = round(float(score), 4)
                dets.append(
                    Detection(bbox=[round(v, 1) for v in box], label=label, score=score, models={self.name: score})
                )
            out.append(dets)
        return out
