import numpy as np
import pytest

from src.models.schemas import Detection
from src.services.detectors.fusion import fuse_detections
from src.services.geometry import (
    apply_transform,
    interpolate_box,
    iou,
    iou_matrix,
    quaternion_to_matrix,
    transform_matrix,
)
from src.services.nuscenes_data import box_corners, points_to_image
from tests.conftest import INTRINSIC


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_matrix_matches_scalar():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    b = np.array([[0, 0, 10, 10], [2, 2, 8, 8], [100, 100, 110, 110]])
    m = iou_matrix(a, b)
    for i in range(2):
        for j in range(3):
            assert m[i, j] == pytest.approx(iou(a[i], b[j]))
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_quaternion_yaw_90():
    r = quaternion_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert r @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0])


def test_transform_inverse_roundtrip():
    q = [0.9, 0.1, -0.2, 0.3]
    m = transform_matrix([1, 2, 3], q)
    inv = transform_matrix([1, 2, 3], q, inverse=True)
    p = np.array([[4.0, -5.0, 6.0]])
    assert apply_transform(inv, apply_transform(m, p)) == pytest.approx(p)


def test_projection_principal_point():
    uv, depth = points_to_image(
        np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -5.0]]), np.eye(4), np.array(INTRINSIC), 1600, 900
    )
    assert len(uv) == 1  # điểm sau camera bị loại
    assert uv[0] == pytest.approx([INTRINSIC[0][2], INTRINSIC[1][2]])
    assert depth[0] == 10.0


def test_box_corners_size():
    c = box_corners([0, 0, 0], [2.0, 4.0, 1.5], [1, 0, 0, 0])
    assert c.max(0) - c.min(0) == pytest.approx([4.0, 2.0, 1.5])


def test_interpolate_box():
    assert interpolate_box([0, 0, 10, 10], 0, [10, 0, 20, 10], 100, 50) == [5, 0, 15, 10]


def det(bbox, label, score, model):
    return Detection(bbox=bbox, label=label, score=score, models={model: score})


def test_fusion_two_models_agree():
    fused = fuse_detections(
        {"a": [det([0, 0, 100, 100], "car", 0.8, "a")], "b": [det([2, 0, 102, 100], "car", 0.6, "b")]},
        0.55,
    )
    assert len(fused) == 1
    f = fused[0]
    assert f.label == "car" and f.score == pytest.approx(0.7)
    assert set(f.models) == {"a", "b"}
    assert 0 < f.bbox[0] < 2


def test_fusion_single_model_score_is_diluted():
    fused = fuse_detections({"a": [det([0, 0, 100, 100], "car", 0.8, "a")], "b": []}, 0.55)
    assert fused[0].score == pytest.approx(0.4)


def test_fusion_class_conflict_becomes_alternative():
    fused = fuse_detections(
        {"a": [det([0, 0, 100, 100], "truck", 0.6, "a"), det([1, 0, 100, 100], "car", 0.5, "a")]},
        0.55,
    )
    assert len(fused) == 1
    assert fused[0].label == "truck"
    assert fused[0].alternatives == {"car": 0.5}


def test_fusion_keeps_separate_objects():
    fused = fuse_detections(
        {"a": [det([0, 0, 100, 100], "car", 0.8, "a"), det([300, 0, 400, 100], "car", 0.7, "a")]}, 0.55
    )
    assert len(fused) == 2
