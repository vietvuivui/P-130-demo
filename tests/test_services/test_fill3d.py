"""Đề xuất box 3D từ box 2D + LiDAR (src/services/fill3d.py, ý tưởng VESPA)."""

from types import SimpleNamespace

import numpy as np

from src.services.bev import GroundModel
from src.services.fill3d import Cam, FillCfg, box_corners, project_bbox, propose

K = np.array([[1000.0, 0, 800], [0, 1000.0, 450], [0, 0, 1]])
CAM_FROM_LIDAR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)  # nhìn theo x


def _scene():
    rng = np.random.default_rng(0)
    ped = np.c_[rng.normal(10, 0.15, 60), rng.normal(0, 0.15, 60), rng.uniform(-1.7, -0.1, 60)]
    road = np.c_[rng.uniform(2, 30, 3000), rng.uniform(-8, 8, 3000), np.full(3000, -1.8)]
    cam = Cam("CAM_FRONT", K, CAM_FROM_LIDAR, 1600, 900, [])
    bbox = project_bbox(box_corners([10, 0, -0.9], [0.7, 0.7, 1.8], 0.0), cam)
    cam.dets = [SimpleNamespace(label="pedestrian", score=0.8, bbox=bbox, mask=None)]
    return np.r_[ped, road], cam


def test_proposes_box_for_object_without_3d_box():
    pts, cam = _scene()
    out = propose(pts, GroundModel([0.0, 0.0, -1.8]), [cam], [], FillCfg())
    assert len(out) == 1
    q = out[0]
    assert q.label == "pedestrian" and np.hypot(q.center[0] - 10.2, q.center[1]) < 0.6
    assert abs(q.center[2] - q.size[2] / 2 - (-1.8)) < 1e-6  # đáy đặt lên mặt đường
    assert q.score == round(0.8 * FillCfg().score_scale, 4)


def test_no_proposal_when_3d_box_explains_detection_or_class_not_allowed():
    pts, cam = _scene()
    gm = GroundModel([0.0, 0.0, -1.8])
    existing = [("pedestrian", np.array([10.0, 0.0, -0.9]), np.array([0.7, 0.7, 1.8]), 0.0)]
    assert propose(pts, gm, [cam], existing, FillCfg()) == []
    assert propose(pts, gm, [cam], [], FillCfg(classes=frozenset({"car"}))) == []
