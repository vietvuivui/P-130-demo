"""BoT-SORT: GMC (affine toàn ảnh), đặc trưng ngoại hình, chi phí ghép, và tracker chọn theo ngoại hình."""

import numpy as np
import pytest

from src.models.schemas import Detection
from src.services import botsort
from src.services.nuscenes_data import TimelineImage
from src.services.propagation import Track, Tracker


def _img(w=320, h=200, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return img


def test_estimate_affine_recovers_shift(tmp_path):
    cv2 = pytest.importorskip("cv2")
    base = cv2.GaussianBlur(_img(640, 400), (3, 3), 0)
    shifted = np.roll(base, (7, -12), axis=(0, 1))  # dịch xuống 7, sang trái 12
    aff = botsort.estimate_affine(
        cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY), 1.0
    )
    assert aff is not None
    box = aff.warp_box([100, 100, 200, 160])
    assert abs(box[0] - 88) < 2 and abs(box[1] - 107) < 2 and abs(box[2] - 188) < 2


def test_embedding_separates_colours():
    pytest.importorskip("cv2")
    red = np.zeros((40, 40, 3), np.uint8)
    red[..., 2] = 220
    red2 = red.copy()
    red2[..., 2] = 200
    blue = np.zeros((40, 40, 3), np.uint8)
    blue[..., 0] = 220
    fr, fr2, fb = (botsort.embed_crop(x) for x in (red, red2, blue))
    assert 1 - fr @ fr2 < 0.1 < 1 - fr @ fb


def test_fuse_cost_gates():
    iou = np.array([[0.8, 0.1]])
    app = np.array([[0.9, 0.05]])
    c = botsort.fuse_cost(iou, app, proximity_thresh=0.5, appearance_thresh=0.25)
    assert c[0, 0] == pytest.approx(0.2)  # ngoại hình quá khác -> chỉ còn IoU
    assert c[0, 1] == pytest.approx(0.9)  # hai box xa nhau -> ngoại hình bị bỏ, chỉ còn 1 - IoU


class _App:
    """Appearance giả: đặc trưng theo nhãn màu của box (toạ độ x1 < 500 = 'đỏ', ngược lại 'xanh')."""

    def features(self, path, boxes):
        out = []
        for b in boxes:
            f = np.array([1.0, 0.0]) if b[0] < 500 else np.array([0.0, 1.0])
            out.append(f)
        return out


def _cfg(config):
    c = config.propagation.model_copy(deep=True)
    c.association, c.flow, c.botsort_gmc = "botsort", "off", False
    return c


def test_tracker_prefers_same_appearance_when_iou_ties(config):
    """Hai detection chồng box dự đoán như nhau; track mang ngoại hình 'đỏ' phải ghép với detection đỏ."""
    cfg = _cfg(config)
    t = Track(track_id="k:1", label="car", box=np.array([100.0, 100.0, 200.0, 160.0]), kind="keep",
              keyframe_id="k", keyframe_object_id="1", last_t=0)  # fmt: skip
    tr = Tracker([t], cfg, 1600, 900, None, "key.jpg", appearance=_App())
    assert t.feat is not None and t.feat[0] == 1.0
    dets = [
        Detection(bbox=[600.0, 100.0, 700.0, 160.0], label="car", score=0.9),  # 'xanh', lệch 500 px
        Detection(bbox=[105.0, 100.0, 205.0, 160.0], label="car", score=0.9),  # 'đỏ', gần
    ]
    tr.step(TimelineImage(sd_token="a", timestamp=100_000, path="a.jpg"), dets)
    assert t.last_match is dets[1]


def test_tracker_appearance_rescues_low_iou(config):
    """IoU dưới ngưỡng nhưng box gần (1 - IoU <= proximity) và cùng ngoại hình -> vẫn ghép (BoT-SORT)."""
    cfg = _cfg(config)
    cfg.match_iou, cfg.botsort_proximity, cfg.botsort_appearance = 0.6, 0.7, 0.3
    t = Track(track_id="k:1", label="car", box=np.array([100.0, 100.0, 200.0, 160.0]), kind="keep",
              keyframe_id="k", keyframe_object_id="1", last_t=0)  # fmt: skip
    tr = Tracker([t], cfg, 1600, 900, None, "key.jpg", appearance=_App())
    det = Detection(bbox=[150.0, 100.0, 250.0, 160.0], label="car", score=0.9)  # IoU 1/3
    tr.step(TimelineImage(sd_token="a", timestamp=100_000, path="a.jpg"), [det])
    assert t.last_match is det
    tr2 = Tracker([Track(track_id="k:2", label="car", box=np.array([100.0, 100.0, 200.0, 160.0]), kind="keep",
                         keyframe_id="k", keyframe_object_id="2", last_t=0)], cfg, 1600, 900, None, None)  # fmt: skip
    tr2.step(TimelineImage(sd_token="a", timestamp=100_000, path="a.jpg"), [det])
    assert tr2.tracks[0].last_match is None  # không có ngoại hình: IoU 1/3 < 0.6 không ghép
