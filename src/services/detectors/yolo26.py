"""YOLO26 (ultralytics, https://github.com/ultralytics/yolo26): detector COCO 80 lớp, không open-vocab.

Chỉ giữ lớp COCO có tên trùng một prompt trong taxonomy (car, truck, bus, person -> pedestrian, bicycle,
motorcycle). Các lớp COCO không có (traffic_cone, barrier, trailer, construction_vehicle) do model open-vocab
trong ensemble lo (YOLOE-26); fusion biết YOLO26 không phủ các lớp đó nên không trừ điểm chúng.
"""

from __future__ import annotations

from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.detectors.base import Detector, pick_device

# Tên lớp COCO của model YOLO pretrained (thứ tự theo chỉ số lớp). Để ở đây để biết YOLO26 phủ lớp nào
# mà không cần nạp torch (fusion khi đọc cache)
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)  # fmt: skip


def coco_class_map(config: AutoLabelConfig, names=COCO_NAMES) -> dict[str, str]:
    """Tên lớp COCO -> lớp nội bộ, qua prompt của taxonomy (khớp đúng tên, không đoán)."""
    prompt_to_class = config.prompt_to_class()
    return {n: prompt_to_class[n.lower()] for n in names if n.lower() in prompt_to_class}


class Yolo26Detector(Detector):
    name = "yolo26"

    def __init__(self, config: AutoLabelConfig):
        super().__init__(config)
        from ultralytics import YOLO

        self.cfg = config.detection.yolo26
        self.device = pick_device()
        self.model = YOLO(self.cfg.weights)
        # Chỉ số lớp của model -> lớp nội bộ (bỏ lớp COCO không thuộc taxonomy)
        mapping = coco_class_map(config, self.model.names.values())
        self.index_to_class = {i: mapping[n] for i, n in self.model.names.items() if n in mapping}

    @property
    def version(self) -> str:
        return f"{self.name}-{Path(self.cfg.weights).stem}-{self.cfg.imgsz}"

    def detect(self, image_paths: list[Path]) -> list[list[Detection]]:
        results = self.model.predict(
            [str(p) for p in image_paths],
            conf=self.config.detection.score_threshold,
            imgsz=self.cfg.imgsz,
            classes=sorted(self.index_to_class),
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
                label = self.index_to_class.get(int(cls_idx))
                if label is None:
                    continue
                score = round(float(conf), 4)
                dets.append(
                    Detection(bbox=[round(v, 1) for v in xyxy], label=label, score=score, models={self.name: score})
                )
            out.append(dets)
        return out
