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


def test_pure_dam4sam_ignores_detections_and_stops_when_mask_is_empty(config, tmp_path, monkeypatch):
    """association = none: box là của SAM, detection không được ghép; mask rỗng mới tính là mất."""
    cfg = config.propagation.model_copy(deep=True)
    cfg.flow, cfg.association, cfg.max_coast_images = "dam4sam", "none", 1

    class Vanishing(FakeSam):
        def _mask(self):
            return np.zeros((900, 1600), np.uint8) if self.n >= 3 else super()._mask()

    pred = Dam4SamPredictor(lambda p: tmp_path / p, tracker_factory=Vanishing)
    monkeypatch.setattr(pred, "image", lambda path: FakeImg())
    t = Track(track_id="k:1", label="car", box=np.array([100.0, 100.0, 200.0, 160.0]), kind="keep",
              keyframe_id="k", keyframe_object_id="1", last_t=0)  # fmt: skip
    tr = Tracker([t], cfg, 1600, 900, None, "k.jpg", predictor=pred)
    far = Detection(bbox=[105.0, 100.0, 205.0, 160.0], label="car", score=0.9)
    for n in (1, 2):
        tr.step(TimelineImage(sd_token=str(n), timestamp=n * 83_333, path=f"{n}.jpg"), [far])
        assert t.last_match is None and t.misses == 0 and t.box[0] == 100.0 + 10 * n
    for n in (3, 4):
        tr.step(TimelineImage(sd_token=str(n), timestamp=n * 83_333, path=f"{n}.jpg"), [far])
    assert not t.alive and "k:1" not in pred._trackers


def test_shared_encoder_runs_once_per_image():
    from src.services.dam4sam import SharedEncoder, _Fast

    runs = []
    enc = SharedEncoder(lambda x: runs.append(x) or {"feat": x})
    with _Fast(enc, "a.jpg", False):
        a1 = enc("img-a")
    with _Fast(enc, "a.jpg", False):
        a2 = enc("img-a")  # vật thứ hai trên cùng ảnh: dùng lại
    with _Fast(enc, "b.jpg", False):
        b = enc("img-b")
    assert a1 is a2 and b == {"feat": "img-b"} and runs == ["img-a", "img-b"] and (enc.hits, enc.misses) == (1, 2)
    assert enc.key is None
    enc("x")
    enc("x")  # không có khoá ảnh: chạy như gốc, không dùng lại
    assert len(runs) == 4


def test_recommend_prefers_simpler_config_within_tie():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("dam4sam_tool", Path(__file__).parents[2] / "tools2d" / "dam4sam.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    base = {"correct": 100, "wrong": 30, "id_switch": 10, "lost": 15, "IDF1": 0.7, "seconds": 15.0}
    res = {
        "flow+byte": {**base, "HOTA": 0.55},
        "dam4sam+none": {**base, "HOTA": 0.61, "seconds": 300.0},
        "dam4sam+byte": {**base, "HOTA": 0.615, "seconds": 310.0},
        "dam4sam+botsort": {**base, "HOTA": 0.618, "seconds": 320.0, "botsort": {"calls": 10, "changed": 1}},
    }
    done = [n for n in tool.COST_ORDER if n in res]
    lines = tool.recommend(res, done)
    assert "`dam4sam+none`" in lines[0]  # 0.61 nằm trong 0.01 của 0.618 và đơn giản hơn
    res["flow+byte"]["HOTA"] = 0.61
    assert "`flow+byte`" in tool.recommend(res, done)[0]
