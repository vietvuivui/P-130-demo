"""Kiểm chứng box 3D bằng camera (src/services/verify3d.py) trên cảnh tổng hợp: 1 camera nhìn thẳng trục x."""

import numpy as np

from src.services.verify3d import (
    V_CLASS,
    V_FP,
    V_MISS,
    V_OK,
    V_SHIFT,
    V_UNSURE,
    Box,
    CamInput,
    points_in_box,
    project_box,
    verify_boxes,
)

K = np.array([[1266.4, 0.0, 800.0], [0.0, 1266.4, 450.0], [0.0, 0.0, 1.0]])
# LiDAR: x trước, y trái, z lên -> camera: x phải, y xuống, z trước
CAM_FROM_LIDAR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)


def car(x=15.0, y=0.0, label="car", score=0.8) -> Box:
    return Box(label, score, np.array([x, y, 0.0]), np.array([1.9, 4.5, 1.6]), 0.0)


def cam(dets, gray=None) -> CamInput:
    if gray is None:
        rng = np.random.default_rng(0)
        gray = rng.integers(40, 220, size=(900, 1600)).astype(np.uint8)  # ảnh "rõ": nhiều chi tiết, đủ sáng
    return CamInput("CAM_FRONT", K, CAM_FROM_LIDAR, 1600, 900, gray, dets)


def cloud_in(box: Box, n=40) -> np.ndarray:
    rng = np.random.default_rng(1)
    w, length, h = box.size
    pts = rng.uniform(
        [-length / 2 + 0.1, -w / 2 + 0.1, -h / 2 + 0.3], [length / 2 - 0.1, w / 2 - 0.1, h / 2 - 0.1], (n, 3)
    )
    return np.c_[pts + box.center, np.zeros(n)]


def det_for(box: Box, label="car", shift=0.0) -> dict:
    b = project_box(box, CAM_FROM_LIDAR, K, 1600, 900)["bbox"]
    dx = shift * (b[2] - b[0])
    return {"bbox": [b[0] + dx, b[1], b[2] + dx, b[3]], "label": label, "score": 0.9, "prompt": label}


def test_projection_and_points_in_box():
    b = car()
    v = project_box(b, CAM_FROM_LIDAR, K, 1600, 900)
    assert v is not None and v["vis"] == 1.0 and 12 < v["depth"] < 18
    assert project_box(car(x=-15), CAM_FROM_LIDAR, K, 1600, 900) is None  # sau lưng camera
    pts = cloud_in(b)
    assert points_in_box(b, pts) == 40
    assert points_in_box(car(x=30), pts) == 0


def test_verdicts():
    b = car()
    pts = cloud_in(b)
    assert verify_boxes([b], [cam([det_for(b)])], pts)[0].verdict == V_OK
    v = verify_boxes([b], [cam([det_for(b, label="truck")])], pts)[0]
    assert v.verdict == V_CLASS and v.level == "high" and v.det_label == "truck"
    assert verify_boxes([b], [cam([det_for(b, shift=0.37)])], pts)[0].verdict == V_SHIFT
    # ảnh rõ, detector không thấy: có điểm LiDAR -> chưa chắc; không có điểm -> nghi báo nhầm
    assert verify_boxes([b], [cam([])], pts)[0].verdict == V_MISS
    v = verify_boxes([b], [cam([])], np.zeros((0, 4)))[0]
    assert v.verdict == V_FP and v.level == "high"
    # ảnh tối -> không đủ thông tin để kết luận
    dark = np.full((900, 1600), 5, np.uint8)
    assert verify_boxes([b], [cam([], gray=dark)], np.zeros((0, 4)))[0].verdict == V_UNSURE


def test_rider_merged_into_bicycle_box():
    bike = Box("bicycle", 0.7, np.array([10.0, 0.0, -0.2]), np.array([0.6, 1.7, 1.7]), 0.0)
    full = project_box(bike, CAM_FROM_LIDAR, K, 1600, 900)["bbox"]
    lower = [full[0], (full[1] + full[3]) / 2, full[2], full[3]]
    upper = [full[0] + 5, full[1], full[2] - 5, (full[1] + full[3]) / 2 + 20]
    dets = [
        {"bbox": lower, "label": "bicycle", "score": 0.8, "prompt": "bicycle"},
        {"bbox": upper, "label": "pedestrian", "score": 0.9, "prompt": "person"},
    ]
    v = verify_boxes([bike], [cam(dets)], cloud_in(bike))[0]
    assert v.verdict == V_OK and v.det_label == "bicycle"


def test_best_threshold_picks_max_f1():
    from src.services.label3d import best_threshold

    pr = {"thresholds": [0.1, 0.3, 0.5], "precision": [0.2, 0.7, 0.95], "recall": [0.9, 0.7, 0.1]}
    assert best_threshold(pr) == 0.3
