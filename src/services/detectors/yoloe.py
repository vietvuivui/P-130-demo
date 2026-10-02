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


def simplify_polygon(poly, eps: float = 1.5, max_points: int = 80) -> list[float] | None:
    """Đa giác mask của Ultralytics (N, 2) -> danh sách phẳng [x1, y1, ...] rút gọn (Douglas-Peucker), để lưu gọn."""
    if poly is None or len(poly) < 3:
        return None
    import cv2

    pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, eps, True)
    while len(approx) > max_points:
        eps *= 1.5
        approx = cv2.approxPolyDP(pts, eps, True)
    if len(approx) < 3:
        return None
    return [round(float(v), 1) for v in approx.reshape(-1)]


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
        v = f"{self.name}-{Path(self.cfg.weights).stem}-{self.cfg.imgsz}" + ("-flip" if self.cfg.tta_flip else "")
        return v + (f"-tiles{self.cfg.tiles}" if self.cfg.tiles > 1 else "")

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
            polys = r.masks.xy if (self.cfg.masks and r.masks is not None) else [None] * len(r.boxes)
            for xyxy, conf, cls_idx, poly in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), polys, strict=True
            ):
                score = round(float(conf), 4)
                dets.append(
                    Detection(
                        mask=simplify_polygon(poly),
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
        if not self.cfg.tta_flip and self.cfg.tiles <= 1:
            return out
        import cv2

        images = [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) for p in image_paths]
        if self.cfg.tta_flip:
            # Thêm lượt ảnh lật ngang, lật box về lại; hai lượt cùng là "yoloe" nên bước gộp (fusion) tự nhập box trùng
            flipped = self._predict([im[:, ::-1].copy() for im in images])
            for dets, fdets, im in zip(out, flipped, images, strict=True):
                w = im.shape[1]
                for d in fdets:
                    x1, y1, x2, y2 = d.bbox
                    d.bbox = [round(w - x2, 1), y1, round(w - x1, 1), y2]
                    if d.mask:
                        d.mask = [round(w - v, 1) if i % 2 == 0 else v for i, v in enumerate(d.mask)]
                dets.extend(fdets)
        if self.cfg.tiles > 1:
            for dets, im in zip(out, images, strict=True):
                dets.extend(self._detect_tiles(im))
        return out

    def _detect_tiles(self, im: np.ndarray, overlap: float = 0.15, margin: int = 4) -> list[Detection]:
        """Chạy model trên lưới ô chồng mép rồi dời box về toạ độ ảnh. Box chạm mép trong của ô (vật bị ô cắt) bị bỏ:
        lượt ảnh đầy đủ đã có box cho vật lớn; ô chỉ để thêm vật nhỏ. Bước gộp của ensemble nhập các box trùng."""
        h, w = im.shape[:2]
        n = self.cfg.tiles
        tw, th = w / n, h / n
        ox, oy = tw * overlap, th * overlap
        crops, rects = [], []
        for r in range(n):
            for c in range(n):
                x1, y1 = int(max(0, c * tw - ox)), int(max(0, r * th - oy))
                x2, y2 = int(min(w, (c + 1) * tw + ox)), int(min(h, (r + 1) * th + oy))
                crops.append(np.ascontiguousarray(im[y1:y2, x1:x2]))
                rects.append((x1, y1, x2, y2))
        found: list[Detection] = []
        for dets, (x1, y1, x2, y2) in zip(self._predict(crops), rects, strict=True):
            for d in dets:
                bx1, by1, bx2, by2 = d.bbox
                cut = ((bx1 <= margin and x1 > 0) or (by1 <= margin and y1 > 0)
                       or (bx2 >= x2 - x1 - margin and x2 < w) or (by2 >= y2 - y1 - margin and y2 < h))  # fmt: skip
                if cut:
                    continue
                d.bbox = [round(bx1 + x1, 1), round(by1 + y1, 1), round(bx2 + x1, 1), round(by2 + y1, 1)]
                if d.mask:
                    d.mask = [round(v + (x1 if i % 2 == 0 else y1), 1) for i, v in enumerate(d.mask)]
                found.append(d)
        return found
