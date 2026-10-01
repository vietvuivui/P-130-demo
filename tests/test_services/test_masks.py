"""FR-04 mask segmentation sơ bộ: rút gọn đa giác, đi qua bước gộp, xuất COCO segmentation."""

import json

import numpy as np

from src.models.schemas import Detection, ReviewActionRequest
from src.services import review
from src.services.detectors.fusion import fuse_detections
from src.services.detectors.yoloe import simplify_polygon
from src.services.exporter import export_dataset
from tests.conftest import make_frame


def test_simplify_polygon():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([100 + 40 * np.cos(t), 80 + 40 * np.sin(t)], 1)
    poly = simplify_polygon(circle)
    assert 6 <= len(poly) // 2 <= 80 and len(poly) % 2 == 0
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) >= 59 and max(xs) <= 141 and min(ys) >= 39 and max(ys) <= 121
    assert simplify_polygon(None) is None and simplify_polygon(np.zeros((2, 2))) is None


def test_fusion_keeps_mask_of_best_member():
    m = [10, 10, 50, 10, 50, 40, 10, 40]
    a = Detection(bbox=[10, 10, 50, 40], label="car", score=0.9, mask=m)
    b = Detection(bbox=[11, 10, 51, 41], label="car", score=0.6)
    fused = fuse_detections({"yoloe": [a], "yolo26": [b]}, 0.5)
    assert len(fused) == 1 and fused[0].mask == m


def test_coco_segmentation(store, config):
    f = make_frame()
    f.objects[0].mask = [100, 400, 300, 400, 300, 520, 100, 520]
    f.objects[1].mask = [700, 420, 760, 420, 760, 560]
    for oid, action in (("1", "KEEP"), ("2", "EDIT_BOX"), ("3", "DELETE")):
        bbox = [705, 425, 765, 565] if action == "EDIT_BOX" else None
        review.apply_action(f, ReviewActionRequest(action=action, object_id=oid, bbox=bbox), "t", config)
    review.approve_frame(f, "t", 10)
    store.save_frame(f)
    res = export_dataset(store, config)
    coco = json.loads((store.exports_dir / res.export_id / "coco.json").read_text())
    segs = {a["object_id"]: a["segmentation"] for a in coco["annotations"]}
    assert segs == {"1": [[100, 400, 300, 400, 300, 520, 100, 520]], "2": []}  # box sửa tay: mask cũ không còn khớp
