"""YOLOE-26 (ultralytics): bản open-vocab của YOLO26, nhận lớp bằng prompt chữ như YOLO-World.

Model phát hành trên Hugging Face: https://huggingface.co/openvision/yoloe26-l-seg. Bản -seg trả cả mask; ở đây
chỉ dùng box. Prompt được mã hoá một lần lúc nạp model bằng text encoder MobileCLIP2 (file TorchScript) theo
đường dẫn trong config, không phụ thuộc thư mục weights toàn cục của ultralytics.
"""

from __future__ import annotations

from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device


class YoloE26Detector(Detector):
    name = "yoloe26"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        from ultralytics import YOLOE
        from ultralytics.nn.text_model import MobileCLIPTS

        self.cfg = config.detection.yoloe26
        self.device = pick_device()
        encoder = Path(self.cfg.text_encoder)
        if not encoder.exists():
            raise FileNotFoundError(f"Thiếu text encoder {encoder}: chạy `python scripts/download_weights.py`")
        self.model = YOLOE(self.cfg.weights)
        # get_text_pe dùng clip_model đã gắn sẵn khi cache_clip_model=True -> nạp encoder từ đúng file trong config
        inner = self.model.model
        inner.clip_model = MobileCLIPTS(next(inner.parameters()).device, weight=str(encoder))
        embeddings = inner.get_text_pe(self.prompts, cache_clip_model=True)
        del inner.clip_model  # chỉ cần lúc mã hoá prompt; bỏ để không giữ ~250 MB trong bộ nhớ
        self.model.set_classes(self.prompts, embeddings)

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
