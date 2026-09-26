"""Florence-2 (unified vision model), task phrase grounding.

Florence-2 không trả confidence nên mọi box nhận `default_score` trong config;
khi chạy cùng model khác, score thật đến từ model kia sau khi fuse.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device

TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2Detector(Detector):
    name = "florence2"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.cfg = config.detection.florence2
        self.device = pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.cfg.model_id, trust_remote_code=True, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.text = ". ".join(self.prompts) + "."

    @property
    def version(self) -> str:
        return f"{self.name}-{self.cfg.model_id.split('/')[-1]}"

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        out = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(text=TASK + self.text, images=image, return_tensors="pt").to(
                self.device, self.dtype
            )
            with self.torch.inference_mode():
                ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
            text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(text, task=TASK, image_size=image.size)[TASK]
            score = self.cfg.default_score
            dets = []
            for box, phrase in zip(parsed["bboxes"], parsed["labels"], strict=True):
                label = self.phrase_to_class(phrase)
                if label is None:
                    continue
                dets.append(
                    Detection(bbox=[round(v, 1) for v in box], label=label, score=score, models={self.name: score})
                )
            out.append(dets)
        return out
