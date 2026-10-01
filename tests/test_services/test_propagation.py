"""Lan truyền nhãn trên video: dữ liệu tổng hợp, không cần GPU hay nuScenes.

Scene giả: 13 ảnh camera 12Hz, keyframe ở ảnh 0, 6, 12 (= frame 000, 001, 002).
- Xe chạy sang phải 10 px mỗi ảnh; detector luôn gọi nó là "car".
- Một "pedestrian" giả (poster) đứng yên, detector thấy ở mọi ảnh.
"""

import pytest

from src.models.schemas import (
    Detection,
    FrameRecord,
    ImageInfo,
    LabelObject,
    QAResult,
    ReviewActionRequest,
)
from src.services import review
from src.services.nuscenes_data import TimelineImage
from src.services.propagation import prop_confidence
from src.services.sequence import PropagationError, evaluate_propagation, propagate_from
from src.services.store import WorkspaceStore

INTRINSIC = [[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]]
SCENE = "scene-0001"
DT = 83_333
KEY_EVERY = 6


def ts(i: int) -> int:
    return 1_000_000 + i * DT


def fid(k: int) -> str:
    return f"{SCENE}_{k:03d}"


def car_box(i: int) -> list[float]:
    x = 100.0 + 10 * i
    return [x, 400.0, x + 200.0, 520.0]


POSTER = [1200.0, 300.0, 1240.0, 420.0]


def default_dets(i: int) -> list[Detection]:
    return [
        Detection(bbox=car_box(i), label="car", score=0.9),
        Detection(bbox=POSTER, label="pedestrian", score=0.45),
    ]


class FakeSource:
    def __init__(self, dets=default_dets, n_images=13, cached=lambda i: True):
        self.dets = dets
        self.n_images = n_images
        self.cached = cached

    def timeline(self, scene, camera):
        return [
            TimelineImage(
                sd_token=f"sd{i}", timestamp=ts(i), sample_token=f"s{i // KEY_EVERY}" if i % KEY_EVERY == 0 else None
            )
            for i in range(self.n_images)
        ]

    def detections(self, sd_token):
        i = int(sd_token[2:])
        return self.dets(i) if self.cached(i) else None


def make_frame(k: int, source: FakeSource, status="auto") -> FrameRecord:
    i = k * KEY_EVERY
    objs = [
        LabelObject(
            object_id=str(n + 1),
            bbox=d.bbox,
            label=d.label,
            score=d.score,
            qa=QAResult(
                risk=0.1, level="low", terms={"detection": 1 - d.score, "lidar": 0.0, "temporal": 0.0, "geometric": 0.0}
            ),
        )
        for n, d in enumerate(source.dets(i))
    ]
    return FrameRecord(
        frame_id=fid(k),
        sample_token=f"s{k}",
        scene=SCENE,
        index=k,
        camera="CAM_FRONT",
        image=ImageInfo(sd_token=f"sd{i}", path=f"samples/CAM_FRONT/{i}.jpg", timestamp=ts(i)),
        intrinsic=INTRINSIC,
        status=status,
        objects=objs,
    )


def approve_keyframe(store: WorkspaceStore, config, actions: list[dict]) -> None:
    """Người duyệt keyframe 000: áp các thao tác, duyệt theo lô phần còn lại, approve."""
    frame = store.load_frame(fid(0))
    for a in actions:
        review.apply_action(frame, ReviewActionRequest(**a), "an", config)
    review.approve_low_risk(frame, "an")
    review.approve_frame(frame, "an", 30)
    store.save_frame(frame)


@pytest.fixture
def scene(tmp_path):
    source = FakeSource()
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    return store, source


def by_label(frame: FrameRecord, label: str) -> list[LabelObject]:
    return [o for o in frame.objects if o.label == label]


# ---- luồng chính ----


def test_propagates_human_decisions_and_follows_motion(scene, config):
    store, source = scene
    # Người sửa lớp xe (car -> truck) và xoá poster bị nhận nhầm là người
    approve_keyframe(
        store,
        config,
        [
            {"action": "CHANGE_CLASS", "object_id": "1", "label": "truck"},
            {"action": "DELETE", "object_id": "2"},
        ],
    )

    resp = propagate_from(store, source, config, fid(0))
    assert resp.frames_updated == [fid(1), fid(2)]
    assert resp.stop_reason == "Hết scene"
    assert resp.objects_propagated == 2 and resp.objects_suppressed == 2
    assert resp.images_without_detections == 0

    key = store.load_frame(fid(0))
    assert key.objects[0].track_id == f"{fid(0)}:1"

    for k in (1, 2):
        f = store.load_frame(fid(k))
        [truck] = by_label(f, "truck")
        assert truck.source == "propagated" and truck.review.status == "pending"
        assert truck.bbox == car_box(k * KEY_EVERY)  # hình học lấy từ detector ở chính frame đó
        assert truck.track_id == f"{fid(0)}:1"
        assert truck.propagation.keyframe_id == fid(0) and truck.propagation.matched
        assert truck.propagation.detector_label == "car"
        assert truck.propagation.prop_conf > 0.85
        # Người đã chốt lớp ở keyframe: lớp detector khác không bị coi là bất thường
        assert [i.code for i in truck.qa.issues] == []
        assert truck.qa.level == "low"

        [poster] = by_label(f, "pedestrian")
        assert poster.review.status == "deleted" and poster.review.action == "PROPAGATED_DELETE"
        assert f.propagated_from == fid(0) and f.status == "auto"
        assert f.prelabel is not None and [o.label for o in f.prelabel] == ["car", "pedestrian"]


def _occluded_car(i):
    # Xe bị che từ ảnh 4: ở keyframe 001 (ảnh 6) chỉ còn box dự đoán; tới ảnh 11 mất dấu quá 6 ảnh thì dừng
    return [Detection(bbox=car_box(i), label="car", score=0.9)] if i < 4 else []


def test_coasting_box_not_written_by_default(tmp_path, config):
    # Mặc định (emit_coasting: false) không ghi box thuần dự đoán; track vẫn chạy rồi dừng như thường
    assert config.propagation.emit_coasting is False
    source = FakeSource(_occluded_car)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [])

    resp = propagate_from(store, source, config, fid(0))
    assert resp.frames_updated == [fid(1)]
    assert resp.tracks_alive == 0
    assert store.load_frame(fid(1)).objects == []


def test_coasting_object_is_flagged_then_stops(tmp_path, config):
    config = config.model_copy(deep=True)
    config.propagation.emit_coasting = True
    source = FakeSource(_occluded_car)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [])

    resp = propagate_from(store, source, config, fid(0))
    assert resp.frames_updated == [fid(1)]
    assert resp.tracks_alive == 0
    assert "dừng lan truyền" in resp.stop_reason

    [car] = store.load_frame(fid(1)).objects
    assert car.source == "propagated" and car.object_id == "p1" and not car.propagation.matched
    codes = {i.code for i in car.qa.issues}
    assert {"PROP_COASTING", "PROP_LOW_CONF"} <= codes
    assert car.qa.level != "low"
    # Box dự đoán tiếp tục theo hướng chuyển động, không đứng yên ở vị trí keyframe
    assert car.bbox[0] > car_box(0)[0]
    assert store.load_frame(fid(2)).propagated_from is None


def test_human_added_box_is_propagated(tmp_path, config):
    # Detector sót cone ở keyframe nhưng thấy ở các ảnh sau
    cone = [800.0, 450.0, 830.0, 500.0]

    def dets(i):
        return [Detection(bbox=cone, label="traffic_cone", score=0.6)] if i > 0 else []

    source = FakeSource(dets)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [{"action": "ADD_BOX", "bbox": [798, 448, 832, 502], "label": "traffic_cone"}])

    propagate_from(store, source, config, fid(0))
    [obj] = store.load_frame(fid(1)).objects
    assert obj.source == "propagated" and obj.label == "traffic_cone" and obj.bbox == cone
    assert obj.propagation.keyframe_object_id == "h1"


def test_stops_before_frame_a_human_opened(scene, config):
    store, source = scene
    f1 = store.load_frame(fid(1))
    f1.status = "editing"
    store.save_frame(f1)
    approve_keyframe(store, config, [])
    before = store.load_frame(fid(1)).model_dump()

    resp = propagate_from(store, source, config, fid(0))
    assert resp.frames_updated == [] and resp.stopped_at == fid(1)
    assert store.load_frame(fid(1)).model_dump() == before


def test_repropagation_starts_from_prelabel(scene, config):
    store, source = scene
    approve_keyframe(store, config, [{"action": "CHANGE_CLASS", "object_id": "1", "label": "truck"}])
    propagate_from(store, source, config, fid(0))
    first = store.load_frame(fid(1))
    propagate_from(store, source, config, fid(0))
    second = store.load_frame(fid(1))

    def strip(f):
        return f.model_dump(exclude={"propagated_at"})

    assert strip(first) == strip(second)
    assert len(second.objects) == 2 and len(second.prelabel) == 2


def test_suppress_needs_same_label(tmp_path, config):
    # Ở frame sau, cùng vị trí poster, detector gọi là barrier: không tự xoá vì có thể là object khác
    def dets(i):
        label = "pedestrian" if i < 6 else "barrier"
        return [Detection(bbox=POSTER, label=label, score=0.6)]

    source = FakeSource(dets)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [{"action": "DELETE", "object_id": "1"}])

    resp = propagate_from(store, source, config, fid(0))
    assert resp.objects_suppressed == 0
    assert "Keyframe không có object nào đã duyệt" in resp.stop_reason
    assert store.load_frame(fid(1)).objects[0].review.status == "pending"


def test_deleted_duplicate_does_not_steal_the_real_object(tmp_path, config):
    # Keyframe có 2 box cho cùng một xe (box thứ hai lệch); người xoá box lệch, giữ box đúng.
    # Frame sau detector chỉ còn 1 box: track "keep" phải giữ được nó.
    def dets(i):
        out = [Detection(bbox=car_box(i), label="car", score=0.9)]
        if i == 0:
            out.append(Detection(bbox=[110.0, 395.0, 300.0, 515.0], label="car", score=0.5))
        return out

    source = FakeSource(dets)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [{"action": "DELETE", "object_id": "2"}])

    resp = propagate_from(store, source, config, fid(0))
    assert resp.frames_updated == [fid(1), fid(2)]
    for k in (1, 2):
        [car] = store.load_frame(fid(k)).objects
        assert car.source == "propagated" and car.propagation.matched and car.review.status == "pending"


def test_only_approved_keyframe_can_propagate(scene, config):
    store, source = scene
    with pytest.raises(PropagationError) as e:
        propagate_from(store, source, config, fid(0))
    assert e.value.code == "NOT_APPROVED" and e.value.status == 409
    with pytest.raises(PropagationError) as e:
        propagate_from(store, source, config, "scene-0001_099")
    assert e.value.code == "FRAME_NOT_FOUND"


def test_images_missing_from_cache_are_predicted_not_counted_as_misses(tmp_path, config):
    # Chỉ keyframe có detection trong cache (chưa chạy detect-sweeps)
    source = FakeSource(cached=lambda i: i % KEY_EVERY == 0)
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    approve_keyframe(store, config, [{"action": "DELETE", "object_id": "2"}])

    resp = propagate_from(store, source, config, fid(0))
    assert resp.images_without_detections == 10
    [car] = [o for o in store.load_frame(fid(1)).objects if o.label == "car"]
    # Xe rộng 200 px, dịch 60 px giữa hai keyframe: vẫn ghép được, không bị coi là mất dấu
    assert car.source == "propagated" and car.propagation.matched


def test_max_frames(scene, config):
    store, source = scene
    approve_keyframe(store, config, [])
    resp = propagate_from(store, source, config, fid(0), max_frames=1)
    assert resp.frames_updated == [fid(1)] and "1 keyframe" in resp.stop_reason


def test_prop_confidence(config):
    cfg = config.propagation
    w = cfg.weights
    full = prop_confidence(0.8, 1.0, 1.0, 1, cfg)
    assert full == pytest.approx((w.agreement * 0.8 + w.continuity + w.lidar) / (w.agreement + w.continuity + w.lidar))
    # Thiếu LiDAR thì chuẩn hoá lại trên các thành phần còn lại
    assert prop_confidence(0.8, 1.0, None, 1, cfg) == pytest.approx(
        (w.agreement * 0.8 + w.continuity) / (w.agreement + w.continuity)
    )
    # Càng xa keyframe gốc càng giảm
    assert prop_confidence(0.8, 1.0, 1.0, 5, cfg) < full
    # Detector không thấy (agreement = 0) thì luôn dưới ngưỡng gắn cờ
    assert prop_confidence(0.0, 1.0, 1.0, 1, cfg) < cfg.flag_below


# ---- số liệu ----


def test_metrics_split_propagated_and_auto_deleted(scene, config):
    store, source = scene
    approve_keyframe(
        store,
        config,
        [
            {"action": "CHANGE_CLASS", "object_id": "1", "label": "truck"},
            {"action": "DELETE", "object_id": "2"},
        ],
    )
    propagate_from(store, source, config, fid(0))
    f1 = store.load_frame(fid(1))
    review.approve_low_risk(f1, "an")
    store.save_frame(f1)

    m = review.compute_metrics(store.list_frames())
    p = m["propagation"]
    assert p["auto_suppressed"] == 2 and p["reviewed"] == 1 and p["fixed"] == 0 and p["pending"] == 1
    assert p["m4_correction_rate"] == 0.0
    # Keyframe: xe bị người đổi lớp, poster bị người xoá -> 2/2 object detector phải sửa
    assert m["model_objects_reviewed"] == 2 and m["m4_correction_rate"] == 1.0


# ---- thí nghiệm keyframe hoàn hảo ----


def test_evaluate_propagation_with_perfect_detector(scene, config):
    store, source = scene
    for k in range(3):
        i = k * KEY_EVERY
        store.save_aux(
            "gt",
            fid(k),
            [
                {"bbox": car_box(i), "label": "car", "instance_token": "inst-car", "ignore": False},
                {"bbox": POSTER, "label": "pedestrian", "instance_token": "inst-ped", "ignore": False},
            ],
        )
    result = evaluate_propagation(store, source, config)
    assert result["scenes"] == 1 and result["tracks_started"] == 2
    assert [r["hop"] for r in result["per_hop"]] == [1, 2]
    for r in result["per_hop"]:
        assert r["correct_rate"] == 1.0 and r["coverage"] == 1.0 and r["id_switch"] == 0
    assert result["calibration"]["n_wrong"] == 0
