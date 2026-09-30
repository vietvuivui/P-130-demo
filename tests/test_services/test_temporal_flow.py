"""Optical flow ở mức box (flow.py), tính lại score theo sweep (temporal_fusion.py), tracker lan truyền dùng flow."""

import numpy as np
import pytest

from src.models.qa_config import PropagationCfg
from src.models.schemas import Detection
from src.services.flow import FlowField, FlowProvider
from src.services.nuscenes_data import TimelineImage
from src.services.propagation import Track, Tracker
from src.services.temporal_fusion import rescore, temporal_score


def texture(h=450, w=800, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    img = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST), (5, 5), 0)


def test_flow_follows_translation(tmp_path):
    import cv2

    base = texture(500, 900)
    a, b = base[20:470, 40:840], base[23:473, 33:833]  # nội dung dịch (+7, -3) px từ a sang b
    cv2.imwrite(str(tmp_path / "a.png"), a)
    cv2.imwrite(str(tmp_path / "b.png"), b)
    field = FlowProvider(lambda p: tmp_path / p, scale=1.0).between("a.png", "b.png")
    for box in ([200, 150, 400, 300], [500, 200, 507, 206]):  # box thường và box nhỏ (chỉ tịnh tiến)
        out = field.warp_box(box)
        assert out == pytest.approx([box[0] + 7, box[1] - 3, box[2] + 7, box[3] - 3], abs=1.0)
    assert FlowProvider(lambda p: tmp_path / p).between("a.png", "missing.png") is None


def test_flow_field_scale_and_edges():
    field = np.zeros((100, 200, 2), np.float32)
    field[..., 0] = np.linspace(-5, 5, 200)[None, :]  # dãn ngang: cạnh trái dịch trái, cạnh phải dịch phải
    out = FlowField(field, 0.5).warp_box([100, 40, 300, 160])
    assert out[0] < 100 and out[2] > 300 and out[1] == pytest.approx(40) and out[3] == pytest.approx(160)


def test_temporal_score_modes():
    assert temporal_score(0.4, [0.5, 0.5, 0.4, 0.4], 4, "mean") == pytest.approx(0.44)
    assert temporal_score(0.45, [], 4, "mean") == pytest.approx(0.09)  # chỉ loé lên ở keyframe
    assert temporal_score(0.2, [0.4, 0.4], 4, "linked") == pytest.approx(1 / 3)
    assert temporal_score(0.3, [0.9], 4, "off") == 0.3
    assert temporal_score(0.3, [], 0, "mean") == 0.3  # không có sweep: giữ nguyên


def test_rescore_uses_warped_boxes_and_same_label():
    key = [Detection(bbox=[100, 100, 200, 200], label="car", score=0.4),
           Detection(bbox=[500, 100, 540, 200], label="pedestrian", score=0.45)]  # fmt: skip
    far = Detection(bbox=[160, 100, 260, 200], label="car", score=0.8)  # vật chạy: lệch 60 px ở sweep
    truck = Detection(bbox=[100, 100, 200, 200], label="truck", score=0.9)
    sweeps = {-1: [far], 1: [truck]}
    no_flow = rescore(key, sweeps, {}, "mean", 0.5)
    assert no_flow[0].score == pytest.approx(0.4 / 3, abs=1e-3)  # IoU 0.25 < 0.5, truck khác lớp: không ghép
    with_flow = rescore(key, sweeps, {-1: [[100, 100, 200, 200]], 1: [truck.bbox]}, "mean", 0.5)
    assert with_flow[0].score == pytest.approx(1.2 / 3, abs=1e-3) and with_flow[0].det_score == 0.4
    assert with_flow[1].score == pytest.approx(0.15, abs=1e-3)  # pedestrian chỉ ở keyframe
    assert rescore(key, sweeps, {}, "off", 0.5) is key


class ShiftFlow:
    def __init__(self, dx):
        self.dx = dx

    def warp_box(self, box):
        return [box[0] + self.dx, box[1], box[2] + self.dx, box[3]]


class FakeMotion:
    def __init__(self, dx):
        self.calls = 0
        self.dx = dx

    def between(self, a, b):
        self.calls += 1
        return ShiftFlow(self.dx)


def _run(flow_mode):
    cfg = PropagationCfg(flow=flow_mode)
    t = Track("t1", "car", np.array([100.0, 400, 300, 520]), "keep", "k", "1", last_t=0)
    motion = FakeMotion(10)
    tracker = Tracker([t], cfg, 1600, 900, motion, "img0.jpg")
    for i in range(1, 6):  # 5 ảnh chưa có detection (vd. ảnh giữa hai keyframe)
        tracker.step(TimelineImage(sd_token=f"sd{i}", timestamp=i * 83_333, path=f"img{i}.jpg"), None)
    return t, tracker


def test_tracker_follows_flow_on_images_without_detections():
    t, tracker = _run("missing")
    assert t.box.tolist() == pytest.approx([150, 400, 350, 520]) and tracker.images_with_flow == 5
    t, tracker = _run("off")  # không flow: vận tốc ban đầu 0, box đứng yên
    assert t.box.tolist() == pytest.approx([100, 400, 300, 520]) and tracker.images_with_flow == 0


def test_sweep_edits_override_detector_for_tracking():
    from src.models.schemas import ReviewState, SweepBox, SweepInfo
    from src.services.sequence import sweep_overrides
    from src.services.sweep_review import effective_detections
    from tests.conftest import make_frame

    frame = make_frame()
    untouched = SweepInfo(offset=1, sd_token="sd1", path="p1.jpg", timestamp=1_100_000,
                          detections=[Detection(bbox=[0, 0, 10, 10], label="car", score=0.5)])  # fmt: skip
    edited = SweepInfo(
        offset=-1, sd_token="sd-1", path="p.jpg", timestamp=900_000,
        detections=[Detection(bbox=[0, 0, 10, 10], label="car", score=0.5)],
        boxes=[SweepBox(box_id="1", bbox=[0, 0, 10, 10], label="car", score=0.5, review=ReviewState(status="deleted")),
               SweepBox(box_id="h1", bbox=[5, 5, 30, 30], label="truck", score=1.0, source="human",
                        review=ReviewState(status="approved", final_bbox=[5, 5, 30, 30], final_label="truck"))],
    )  # fmt: skip
    frame.sweeps = [edited, untouched]
    dets = effective_detections(edited)
    assert [(d.label, d.score) for d in dets] == [("truck", 1.0)]
    assert effective_detections(untouched) == untouched.detections
    assert set(sweep_overrides([frame])) == {"sd-1"}  # sweep chưa sửa vẫn đọc cache detector


def test_byte_association_prefers_confident_boxes():
    """Box score thấp nằm sát hơn không được giành track của vật có box rõ (ghép hai lượt kiểu ByteTrack)."""
    track_box = np.array([100.0, 100, 200, 200])
    confident = Detection(bbox=[112, 100, 212, 200], label="car", score=0.9)  # IoU ~0.79 với dự đoán
    faint = Detection(bbox=[102, 100, 202, 200], label="car", score=0.15)  # IoU ~0.96 nhưng score thấp
    img = TimelineImage(sd_token="sd1", timestamp=83_333, path="")
    out = {}
    for assoc in ("single", "byte"):
        t = Track("t", "car", track_box.copy(), "keep", "k", "1", last_t=0)
        Tracker([t], PropagationCfg(flow="off", association=assoc), 1600, 900).step(img, [faint, confident])
        out[assoc] = t.last_match.score
    assert out == {"single": 0.15, "byte": 0.9}
