"""OC-SORT trong tracker lan truyền 2D và 3D (OCR, ORU, OCM)."""

import numpy as np

from src.models.qa_config import Propagation3DCfg, PropagationCfg
from src.models.schemas import Detection
from src.services.nuscenes_data import TimelineImage
from src.services.propagation import Track, Tracker
from src.services.propagation3d import Det3D, Track3D, Tracker3D

DT = 0.083333


def _img(i: int) -> TimelineImage:
    return TimelineImage(sd_token=f"s{i}", timestamp=int(i * DT * 1e6))


def _det(box, score=0.9, label="car") -> Detection:
    return Detection(bbox=box, label=label, score=score, detector="test")


def _run(cfg: PropagationCfg, hidden: range, n: int = 20) -> Track:
    """Xe đi sang phải 5 px/ảnh, bị che (không có detection) ở các ảnh `hidden`."""
    t = Track("t1", "car", np.array([100.0, 100, 200, 180]), "keep", "k0", "o1", last_t=0)
    tr = Tracker([t], cfg, 1600, 900)
    for i in range(1, n):
        box = [100.0 + 5 * i, 100, 200.0 + 5 * i, 180]
        tr.step(_img(i), [] if i in hidden else [_det(box)])
    return t


def test_ocr_keeps_track_through_long_occlusion():
    assert not _run(PropagationCfg(flow="off"), range(3, 13)).alive  # mất 10 ảnh > max_coast_images (6): dừng
    t = _run(PropagationCfg(flow="off", oc_recover=True, oc_max_lost=18), range(3, 13))
    assert t.alive and t.last_match is not None


def test_oru_velocity_from_observations_after_gap():
    t = _run(PropagationCfg(flow="off", oc_recover=True, oc_reupdate=True), range(3, 9), n=10)
    # quan sát cuối trước khi bị che: ảnh 2; nhận lại ở ảnh 9 -> vận tốc = 35 px / 7 ảnh = 60 px/s
    assert np.allclose(t.vel[[0, 2]], 5 / DT, rtol=0.02)


def test_ocm_prefers_detection_along_observed_direction():
    cfg = PropagationCfg(flow="off", oc_momentum=0.5, smoothing=1.0)
    t = Track("t1", "car", np.array([100.0, 100, 200, 180]), "keep", "k0", "o1", last_t=0)
    tr = Tracker([t], cfg, 1600, 900)
    for i in range(1, 5):  # đi sang phải 10 px/ảnh
        tr.step(_img(i), [_det([100.0 + 10 * i, 100, 200.0 + 10 * i, 180])])
    # hai detection lệch đều hai phía box dự đoán: một phía trước (cùng hướng đi), một phía sau
    x1 = t.box[0] + t.vel[0] * DT
    ahead, behind = [x1 + 12, 100, x1 + 112, 180], [x1 - 12, 100, x1 + 88, 180]
    tr.step(_img(5), [_det(behind), _det(ahead)])
    assert t.last_match.bbox == ahead


def test_ocr_3d_recovers_by_last_observation():
    cfg = Propagation3DCfg(oc_recover=True, oc_max_lost=4)
    t = Track3D("a", "car", [2, 4.5, 1.7], np.array([0.0, 0, 0]), 0.0, np.array([10.0, 0]), "keep")
    tr = Tracker3D([t], cfg)
    tr.step([Det3D(np.array([5.0, 0.0, 0.0]), 0.0, np.array([10.0, 0]), "car", 0.9)], 0.5)
    tr.step([], 0.5)  # bị che
    t.vel = np.array([30.0, 0])  # vận tốc hỏng: dự đoán trôi xa, lượt ghép chính không tới
    tr.step([Det3D(np.array([15.0, 0.0, 0.0]), 0.0, np.array([10.0, 0]), "car", 0.9)], 0.5)
    assert t.alive and t.recovered == 1 and np.allclose(t.center[:2], [15, 0])
