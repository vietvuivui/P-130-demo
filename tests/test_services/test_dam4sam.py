"""Dam4SamPredictor với tracker giả (không cần GPU / SAM 2.1): mask -> box, một tracker cho mỗi track, nối vào Tracker."""

import numpy as np
import pytest

from src.models.schemas import Detection
from src.services.dam4sam import Dam4SamPredictor, mask_to_box
from src.services.nuscenes_data import TimelineImage
from src.services.propagation import Track, Tracker


class FakeSam:
    """Giả DAM4SAMTracker: vật đi sang phải 10 px mỗi ảnh (theo thứ tự ảnh được đưa vào), mask là box đó."""

    def __init__(self):
        self.box = None
        self.n = 0

    def initialize(self, image, init_mask, bbox=None):
        self.box = list(bbox)
        return {"pred_mask": self._mask()}

    def track(self, image):
        self.n += 1
        return {"pred_mask": self._mask()}

    def _mask(self):
        x, y, w, h = self.box
        m = np.zeros((900, 1600), np.uint8)
        x = int(x + 10 * self.n)
        m[int(y) : int(y + h), x : int(x + w)] = 1
        return m


class FakeImg:
    pass


def test_mask_to_box():
    m = np.zeros((20, 30), np.uint8)
    assert mask_to_box(m) is None
    m[5:10, 3:12] = 1
    assert mask_to_box(m) == [3.0, 5.0, 12.0, 10.0]


def test_predictor_tracks_each_object(tmp_path, monkeypatch):
    pred = Dam4SamPredictor(lambda p: tmp_path / p, tracker_factory=FakeSam)
    monkeypatch.setattr(pred, "image", lambda path: FakeImg())  # không đọc file
    assert pred.start("t1", "k.jpg", [100, 100, 200, 160])
    assert pred.start("t2", "k.jpg", [500, 100, 600, 160])
    assert pred.predict("t1", "a.jpg") == [110.0, 100.0, 210.0, 160.0]
    assert pred.predict("t2", "a.jpg") == [510.0, 100.0, 610.0, 160.0]
    assert pred.predict("t1", "b.jpg") == [120.0, 100.0, 220.0, 160.0]
    pred.stop("t1")
    assert pred.predict("t1", "c.jpg") is None and pred.calls == 3


def test_tracker_uses_sam_box_when_detector_misses(config, tmp_path, monkeypatch):
    cfg = config.propagation.model_copy(deep=True)
    cfg.flow = "dam4sam"
    pred = Dam4SamPredictor(lambda p: tmp_path / p, tracker_factory=FakeSam)
    monkeypatch.setattr(pred, "image", lambda path: FakeImg())
    t = Track(track_id="k:1", label="car", box=np.array([100.0, 100.0, 200.0, 160.0]), kind="keep",
              keyframe_id="k", keyframe_object_id="1", last_t=0)  # fmt: skip
    tr = Tracker([t], cfg, 1600, 900, None, "k.jpg", predictor=pred)
    assert tr.predictor is pred and "k:1" in pred._trackers
    # Ảnh 1: detector sót -> box theo SAM (dời 10 px), không phải đứng yên như dự đoán vận tốc 0
    tr.step(TimelineImage(sd_token="a", timestamp=83_333, path="a.jpg"), [])
    assert t.box.tolist() == [110.0, 100.0, 210.0, 160.0] and t.misses == 1
    # Ảnh 2: detector thấy (lệch 1 px so với SAM) -> ghép được, box làm mượt về phía detection
    det = Detection(bbox=[121.0, 100.0, 221.0, 160.0], label="car", score=0.9)
    tr.step(TimelineImage(sd_token="b", timestamp=166_666, path="b.jpg"), [det])
    assert t.last_match is det and t.misses == 0 and tr.images_with_sam == 2
    # Không có flow khi dùng dam4sam
    assert tr.motion is None


def test_predictor_without_repo_gives_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="DAM4SAM"):
        Dam4SamPredictor(lambda p: tmp_path / p, repo_dir=tmp_path / "nope")
