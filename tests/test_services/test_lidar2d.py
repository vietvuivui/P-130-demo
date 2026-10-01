"""Box 3D chiếu xuống ảnh và gộp với box detector 2D (src/services/lidar2d.py)."""

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection
from src.services.lidar2d import SOURCE, merge_boxes, merge_detections, project_boxes

K = np.array([[1000.0, 0, 800], [0, 1000.0, 450], [0, 0, 1]])
CAM = np.eye(4)  # hệ toàn cục trùng hệ camera: z hướng ra trước


def box3d(x, z, score=0.8, name="car", size=(2.0, 4.0, 1.5)):
    return {"translation": [x, 0.0, z], "size": list(size), "rotation": [1, 0, 0, 0], "detection_name": name,
            "detection_score": score}  # fmt: skip


def test_project_boxes_bounding_rect_and_filters():
    out = project_boxes([box3d(0, 20), box3d(0, -20), box3d(0, 20, score=0.01), box3d(500, 20)], CAM, K, 1600, 900)
    assert len(out) == 1  # sau lưng camera, điểm thấp, ngoài khung ảnh đều bị bỏ
    b = out[0]
    assert b["label"] == "car" and b["score"] == 0.8
    # box dài 4 m theo trục x, cao 2 m (trục y), mặt gần ở z = 19.25
    assert np.allclose(
        b["bbox"], [800 - 2000 / 19.25, 450 - 1000 / 19.25, 800 + 2000 / 19.25, 450 + 1000 / 19.25], atol=0.1
    )


def test_project_boxes_clips_to_image():
    (b,) = project_boxes([box3d(-14, 20)], CAM, K, 1600, 900)
    assert b["bbox"][0] == 0.0 and 0 < b["bbox"][2] < 400


def test_merge_boxes_three_cases():
    b3 = [{"bbox": [100, 100, 200, 200], "label": "car", "score": 0.6},
          {"bbox": [500, 100, 600, 200], "label": "pedestrian", "score": 0.4}]  # fmt: skip
    b2 = [{"bbox": [105, 105, 200, 200], "label": "car", "score": 0.9},  # trùng box 3D
          {"bbox": [500, 100, 600, 200], "label": "car", "score": 0.7},  # trùng vị trí nhưng khác lớp: không ghép
          {"bbox": [900, 100, 950, 200], "label": "truck", "score": 0.8}]  # fmt: skip
    out = merge_boxes(b3, b2, match_iou=0.5, camera_only_scale=0.5)
    assert [(p is not None, j) for p, j, _ in out] == [(True, 0), (True, None), (False, 1), (False, 2)]
    assert [round(s, 2) for _, _, s in out] == [0.9, 0.4, 0.35, 0.4]


def test_merge_detections_keeps_3d_box_and_marks_source():
    dets = [Detection(bbox=[105, 105, 200, 200], label="car", score=0.9, models={"yoloe": 0.9}, mask=[1, 1, 2, 2, 3, 3],
                      alternatives={"truck": 0.3}),
            Detection(bbox=[900, 100, 950, 200], label="truck", score=0.8, models={"yoloe": 0.8})]  # fmt: skip
    b3 = [{"bbox": [100, 100, 200, 200], "label": "car", "score": 0.6},
          {"bbox": [500, 100, 600, 200], "label": "pedestrian", "score": 0.4}]  # fmt: skip
    out = merge_detections(dets, b3, 0.5, 0.5)
    assert [d.label for d in out] == [
        "car",
        "pedestrian",
        "truck",
    ]  # sắp theo điểm giảm dần (truck còn 0.4, sau 3D 0.4)
    car, ped, truck = out
    assert car.bbox == [100, 100, 200, 200] and car.score == 0.9 and car.mask is None
    assert car.models == {"yoloe": 0.9, SOURCE: 0.6} and car.alternatives == {"truck": 0.3}
    assert ped.models == {SOURCE: 0.4}
    assert truck.bbox == [900, 100, 950, 200] and truck.score == 0.4 and truck.det_score == 0.8


def test_no_3d_boxes_only_scales_camera_scores():
    dets = [Detection(bbox=[0, 0, 10, 10], label="car", score=0.5)]
    (d,) = merge_detections(dets, [], 0.5, 0.5)
    assert d.score == 0.25 and d.det_score == 0.5


def test_config_defaults():
    c = AutoLabelConfig().detection.lidar3d
    assert c.enabled and c.camera_only_scale == 0.5 and c.match_iou == 0.5
