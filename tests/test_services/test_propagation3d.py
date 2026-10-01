"""Lan truyền box 3D đã duyệt (propagation3d.py): bù chuyển động xe, dịch theo vận tốc, khoá lớp / kích thước người."""

import numpy as np
import pytest

from src.models.qa_config import Propagation3DCfg
from src.models.schemas import ReviewState
from src.models.schemas3d import Box3D, Frame3DRecord, Object3D, Verify3D
from src.services.propagation3d import propagate3d_from, to_global, to_lidar
from src.services.store import WorkspaceStore


def pose(x: float, yaw: float = 0.0) -> list[list[float]]:
    c, s = np.cos(yaw), np.sin(yaw)
    return [[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def obj(oid, center, label="car", vel=(0.0, 0.0), size=(1.9, 4.5, 1.6), score=0.8):
    return Object3D(object_id=oid, label=label, score=score, box=Box3D(center=list(center), size=list(size), yaw=0.0,
                    velocity=list(vel)), verify=Verify3D(verdict="DUNG", level="low", comment=""))  # fmt: skip


def frame(i, x_ego, objects, status="auto"):
    return Frame3DRecord(frame_id=f"scene-0001_{i:03d}", model="m", sample_token=f"s{i}", scene="scene-0001", index=i,
                         lidar_sd_token=f"l{i}", global_from_lidar=pose(x_ego), objects=objects, status=status)  # fmt: skip


def test_global_roundtrip():
    b = Box3D(center=[3.0, 1.0, 0.5], size=[1, 2, 1], yaw=0.3, velocity=[1.0, 0.5])
    g = pose(10.0, 0.7)
    c, yaw, v = to_global(b, g)
    back = to_lidar(c, yaw, v, b.size, g)
    assert back.center == pytest.approx(b.center, abs=1e-3) and back.yaw == pytest.approx(0.3, abs=1e-3)
    assert back.velocity == pytest.approx([1.0, 0.5], abs=1e-3)


@pytest.fixture
def scene(tmp_path):
    store = WorkspaceStore(tmp_path / "ws")
    # Xe tự lái đi 5 m mỗi keyframe; xe A chạy cùng chiều 10 m/s (trong hệ LiDAR lùi dần về phía trước 5 m / 0.5 s),
    # cone đứng yên (trong hệ LiDAR lùi 5 m mỗi frame). Người: A đổi lớp thành truck + sửa kích thước; xoá box "B" sai.
    k0 = frame(0, 0.0, [obj("1", (10, 0, 0), vel=(10, 0)), obj("2", (20, 5, 0), "traffic_cone"), obj("3", (15, -8, 0))],
               status="approved")  # fmt: skip
    k0.objects[0].box = k0.objects[0].box.model_copy(update={"size": [2.5, 8.0, 3.0]})
    k0.objects[0].review = ReviewState(status="approved", action="EDIT_BOX", final_label="truck")
    k0.objects[1].review = ReviewState(status="approved", action="KEEP", final_label="traffic_cone")
    k0.objects[2].review = ReviewState(status="deleted", action="DELETE")
    store.save_frame3d(k0)
    # frame 1: A ở global x=15 -> LiDAR x=10; cone global 20 -> LiDAR 15; box sai vẫn ở global 15 -> LiDAR 10, y=-8
    store.save_frame3d(frame(1, 5.0, [obj("1", (10.3, 0, 0), vel=(10, 0)), obj("2", (15.1, 5, 0), "traffic_cone"),
                                      obj("3", (10, -8, 0)), obj("4", (30, 10, 0))]))  # fmt: skip
    store.save_frame3d(frame(2, 10.0, [obj("1", (10, 0, 0), vel=(10, 0))]))
    store.save_frame3d(frame(3, 15.0, [obj("1", (10, 0, 0))], status="editing"))  # người đã mở: dừng trước
    return store


def test_propagate3d(scene):
    r = propagate3d_from(scene, "m", "scene-0001_000", Propagation3DCfg())
    assert r["frames_updated"] == ["scene-0001_001", "scene-0001_002"] and "Dừng trước scene-0001_003" in r["stop_reason"]
    f1 = scene.load_frame3d("m", "scene-0001_001")
    by = {o.object_id: o for o in f1.objects}
    a = by["1"]
    assert a.source == "propagated" and a.label == "truck" and a.box.size == [2.5, 8.0, 3.0]
    assert a.box.center[0] == pytest.approx(10.3) and a.review.status == "pending" and a.track_id == "scene-0001_000:1"
    assert a.propagation.keyframe_id == "scene-0001_000" and a.propagation.distance_m == pytest.approx(0.3, abs=0.01)
    assert a.verify.level == "medium"  # mô hình ở frame này gọi là car, người chốt truck -> người xem lại
    assert by["2"].source == "propagated" and by["2"].verify.level == "low"
    assert by["3"].review.action == "PROPAGATED_DELETE"  # box sai người đã xoá bị tự xoá
    assert by["4"].source == "model" and by["4"].propagation is None  # vật mới không đụng tới
    assert f1.prelabel is not None and f1.propagated_from == "scene-0001_000"
    assert scene.load_frame3d("m", "scene-0001_003").objects[0].source == "model"
    # Lan truyền lại không cộng dồn (dựng từ prelabel)
    propagate3d_from(scene, "m", "scene-0001_000", Propagation3DCfg())
    assert len(scene.load_frame3d("m", "scene-0001_001").objects) == 4


def test_propagate3d_without_velocity_loses_moving_object(scene):
    propagate3d_from(scene, "m", "scene-0001_000", Propagation3DCfg(motion="none"))
    by = {o.object_id: o for o in scene.load_frame3d("m", "scene-0001_001").objects}
    assert by["1"].source == "model"  # chỉ bù chuyển động xe: xe A đã chạy 5 m, ngoài ngưỡng khớp
    assert by["2"].source == "propagated"  # vật đứng yên vẫn khớp


def test_propagate3d_requires_approved(scene):
    with pytest.raises(ValueError):
        propagate3d_from(scene, "m", "scene-0001_001", Propagation3DCfg())
