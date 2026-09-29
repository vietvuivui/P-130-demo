"""YOLOE-26 (ultralytics): open-vocab trên nền YOLO26, prompt bằng chữ như YOLO-World.

Text encoder là MobileCLIP2 (tải tự động từ GitHub của Ultralytics), không cần CLIP của OpenAI.
Model -seg trả cả mask; ở đây chỉ dùng box. Nhận cả trọng số đã fine-tune trên nuImages (tools2d/finetune_yoloe.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
        # Trọng số đã fine-tune (tools2d/finetune_yoloe.py) là tập lớp đóng, tên lớp = tên lớp của config ("traffic cone"):
        # dùng thẳng, không đặt prompt chữ (đầu phân lớp đã học xong, prompt mới sẽ ghi đè mất)
        names = {int(i): str(n).replace(" ", "_") for i, n in (self.model.names or {}).items()}
        self.fixed_labels = names if names and set(names.values()) <= set(config.classes) else None
        if self.fixed_labels is None:
            self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

    @property
    def version(self) -> str:
        return f"{self.name}-{Path(self.cfg.weights).stem}-{self.cfg.imgsz}" + ("-flip" if self.cfg.tta_flip else "")

    def _predict(self, sources: list) -> list[list[Detection]]:
        results = self.model.predict(
            sources,
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
                        label=self.fixed_labels[int(cls_idx)]
                        if self.fixed_labels
                        else self.prompt_to_class[self.prompts[int(cls_idx)]],
                        score=score,
                        models={self.name: score},
                    )
                )
            out.append(dets)
        return out

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        out = self._predict([str(p) for p in image_paths])
        if not self.cfg.tta_flip:
            return out
        # Thêm lượt ảnh lật ngang, lật box về lại; hai lượt cùng là "yoloe" nên bước gộp (fusion) tự nhập box trùng
        import cv2

        images = [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) for p in image_paths]
        flipped = self._predict([im[:, ::-1].copy() for im in images])
        for dets, fdets, im in zip(out, flipped, images, strict=True):
            w = im.shape[1]
            for d in fdets:
                x1, y1, x2, y2 = d.bbox
                d.bbox = [round(w - x2, 1), y1, round(w - x1, 1), y2]
            dets.extend(fdets)
        return out
