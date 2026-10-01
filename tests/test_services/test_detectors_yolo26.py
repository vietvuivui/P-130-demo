"""Ensemble YOLO26 (COCO) + YOLOE-26 (open-vocab): ánh xạ lớp và fusion theo lớp mỗi model phủ. Không cần torch."""

import pytest

from src.models.schemas import Detection
from src.services.detectors import DetectorEnsemble, detector_classes
from src.services.detectors.fusion import fuse_detections
from src.services.detectors.yolo26 import coco_class_map

BOX = [100.0, 100.0, 200.0, 180.0]


def det(model, label, score, box=BOX):
    return Detection(bbox=box, label=label, score=score, models={model: score})


def test_coco_classes_map_to_taxonomy(config):
    assert coco_class_map(config) == {
        "person": "pedestrian",
        "bicycle": "bicycle",
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "truck": "truck",
    }
    # 4 lớp COCO không có: YOLO26 không phủ, YOLOE-26 (open-vocab) phủ hết
    assert detector_classes("yolo26", config).isdisjoint({"traffic_cone", "barrier", "trailer", "construction_vehicle"})
    assert detector_classes("yoloe26", config) is None


def test_default_detector_and_coverage(config, tmp_path):
    assert config.detection.detectors == ["yoloe"]
    assert DetectorEnsemble(config, tmp_path).coverage == {"yoloe": None}  # không nạp model
    ens = DetectorEnsemble(config, tmp_path, ["yolo26", "yoloe26"])
    assert ens.coverage["yoloe26"] is None and "car" in ens.coverage["yolo26"]
    # `detect-sweeps --detectors` đổi names sau khi tạo ensemble: coverage phải đi theo
    ens.names = ["yolo26"]
    assert set(ens.coverage) == {"yolo26"}


def test_fusion_does_not_penalise_classes_a_model_cannot_see(config, tmp_path):
    coverage = DetectorEnsemble(config, tmp_path, ["yolo26", "yoloe26"]).coverage
    # Cone chỉ YOLOE thấy: YOLO26 không có lớp cone nên không bị tính là phiếu chống
    [cone] = fuse_detections({"yolo26": [], "yoloe26": [det("yoloe26", "traffic_cone", 0.6)]}, 0.55, coverage)
    assert cone.score == pytest.approx(0.6)
    # Car chỉ YOLOE thấy: YOLO26 nhận được car mà không thấy -> điểm bị chia đôi như trước
    [car] = fuse_detections({"yolo26": [], "yoloe26": [det("yoloe26", "car", 0.6)]}, 0.55, coverage)
    assert car.score == pytest.approx(0.3)
    # Cả hai cùng thấy car
    [both] = fuse_detections(
        {"yolo26": [det("yolo26", "car", 0.9)], "yoloe26": [det("yoloe26", "car", 0.7)]}, 0.55, coverage
    )
    assert both.score == pytest.approx(0.8) and both.models == {"yolo26": 0.9, "yoloe26": 0.7}


def test_fusion_without_coverage_is_unchanged():
    [d] = fuse_detections({"a": [], "b": [det("b", "traffic_cone", 0.6)]}, 0.55)
    assert d.score == pytest.approx(0.3)
