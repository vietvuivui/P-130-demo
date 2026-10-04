"""API phần 3D: danh sách, frame, point cloud, thao tác duyệt, approve, số liệu, xuất nhãn."""

import json

import numpy as np
import pytest

from src.models.schemas3d import Box3D, Camera3D, Frame3DRecord, Object3D, Verify3D
from src.services.eval3d import evaluate_verification
from src.services.review3d import _matrix_to_quaternion
from src.services.store import WorkspaceStore

CAM = Camera3D(
    sd_token="cam0", path="samples/CAM_FRONT/x.jpg", intrinsic=[[1266.4, 0, 800], [0, 1266.4, 450], [0, 0, 1]],
    cam_from_lidar=[[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
)  # fmt: skip


def obj(oid, level, verdict, label="car", x=10.0):
    return Object3D(
        object_id=oid, label=label, score=0.7, box=Box3D(center=[x, 0, 0], size=[1.9, 4.5, 1.6], yaw=0.3),
        verify=Verify3D(verdict=verdict, level=level, comment="c", camera="CAM_FRONT", distance_m=x),
    )  # fmt: skip


def frame3d(fid="scene-0035_000"):
    yaw = 0.5
    g = np.eye(4)
    g[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    g[:3, 3] = [100, 200, 1]
    return Frame3DRecord(
        frame_id=fid, model="pointpillars", sample_token="tok-" + fid, scene="scene-0035", index=0,
        lidar_sd_token="lid0", global_from_lidar=g.tolist(), cameras={"CAM_FRONT": CAM},
        objects=[obj("1", "low", "DUNG"), obj("2", "high", "NGHI BAO NHAM", x=20), obj("3", "high", "SAI LOP", x=30)],
    )  # fmt: skip


@pytest.fixture
def store3d(store: WorkspaceStore) -> WorkspaceStore:
    """Thêm một frame 3D vào workspace tạm mà fixture `client` đang dùng."""
    store.save_frame3d(frame3d())
    path = store.points_path("scene-0035_000")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.zeros((10, 4), np.float32).tofile(path)
    store.save_aux("gt3d", "scene-0035_000", [
        {"label": "car", "center": [10.3, 0, 0], "size": [1.9, 4.5, 1.6], "yaw": 0.3, "num_pts": 20},
        {"label": "truck", "center": [30.2, 0.1, 0], "size": [2.5, 8, 3], "yaw": 0.0, "num_pts": 50},
        {"label": "pedestrian", "center": [5, 5, 0], "size": [0.6, 0.6, 1.7], "yaw": 0.0, "num_pts": 8},
    ])  # fmt: skip
    return store


@pytest.mark.asyncio
async def test_review_flow_and_export(client, store3d):
    r = await client.get("/api/v1/3d/models")
    assert r.status_code == 200 and r.json()[0]["model"] == "pointpillars"
    r = await client.get("/api/v1/3d/frames", params={"model": "pointpillars"})
    s = r.json()[0]
    assert s["counts"] == {"low": 1, "medium": 0, "high": 2} and s["pending"] == 3
    r = await client.get("/api/v1/3d/frames/pointpillars/scene-0035_000/points")
    assert r.status_code == 200 and len(r.content) == 10 * 16

    base = "/api/v1/3d/frames/pointpillars/scene-0035_000"
    assert (await client.post(base + "/approve", json={})).status_code == 409  # còn box chờ duyệt
    assert (await client.post(base + "/actions", json={"action": "DELETE", "object_id": "2"})).status_code == 200
    r = await client.post(base + "/actions", json={"action": "CHANGE_CLASS", "object_id": "3", "label": "truck"})
    assert r.json()["objects"][2]["review"]["final_label"] == "truck"
    bad = await client.post(base + "/actions", json={"action": "CHANGE_CLASS", "object_id": "3", "label": "ufo"})
    assert bad.status_code == 422
    r = await client.post(base + "/approve-low-risk", json={})
    assert r.json()["objects"][0]["review"]["action"] == "BATCH_APPROVE"
    r = await client.post(base + "/approve", json={"review_time_s": 12})
    assert r.json()["status"] == "approved"

    m = (await client.get("/api/v1/3d/metrics", params={"model": "pointpillars"})).json()
    assert m["approved"] == 1 and m["fixed"] == 2 and m["flag_precision"] == 1.0 and m["flag_recall"] == 1.0
    assert m["by_verdict"]["DUNG"] == {"reviewed": 1, "fixed": 0}

    r = await client.post("/api/v1/3d/export", params={"model": "pointpillars"})
    assert r.status_code == 200 and r.json()["objects"] == 2
    out = json.loads((store3d.root / "exports").glob("3d-*").__next__().joinpath("labels3d_nusc.json").read_text())
    rows = out["results"]["tok-scene-0035_000"]
    assert {b["detection_name"] for b in rows} == {"car", "truck"}
    # box car ở (10, 0) hệ LiDAR, xe quay 0.5 rad, dịch (100, 200) -> hệ toàn cục
    car = next(b for b in rows if b["detection_name"] == "car")
    assert car["translation"][:2] == pytest.approx([100 + 10 * np.cos(0.5), 200 + 10 * np.sin(0.5)], abs=1e-3)


def test_quaternion_roundtrip():
    for yaw in (0.0, 1.0, 3.0, -2.5):
        c, s = np.cos(yaw), np.sin(yaw)
        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        q = _matrix_to_quaternion(m)
        assert abs(np.linalg.norm(q) - 1) < 1e-5


def test_verification_vs_gt(store3d):
    res = evaluate_verification(store3d, "pointpillars")
    # box 1 đúng (car gần GT car), box 2 báo nhầm (không GT nào gần), box 3 sai lớp (GT truck)
    assert res["table"]["DUNG"]["dung"] == 1
    assert res["table"]["NGHI BAO NHAM"]["bao nham"] == 1
    assert res["table"]["SAI LOP"]["sai lop"] == 1
    assert res["auto_precision"] == 1.0 and res["error_recall"] == 1.0 and res["gt_missed"] == 2


@pytest.mark.asyncio
async def test_add_and_edit_box(client, store3d):
    base = "/api/v1/3d/frames/pointpillars/scene-0035_000"
    box = {"center": [5.0, 5.0, -0.9], "size": [0.7, 0.7, 1.8], "yaw": 7.0}
    r = await client.post(base + "/actions", json={"action": "ADD_BOX", "label": "pedestrian", "box": box})
    assert r.status_code == 200
    new = r.json()["objects"][-1]
    assert new["object_id"] == "4" and new["source"] == "human" and new["review"]["action"] == "ADD_BOX"
    assert new["box"]["yaw"] == pytest.approx(7.0 - 2 * np.pi, abs=1e-4)  # chuẩn hoá về [-pi, pi]
    bad = {"center": [0, 0, 0], "size": [0.0, 1, 1], "yaw": 0}
    assert (
        await client.post(base + "/actions", json={"action": "ADD_BOX", "label": "car", "box": bad})
    ).status_code == 422
    assert (
        await client.post(base + "/actions", json={"action": "ADD_BOX", "box": box})
    ).status_code == 422  # thiếu lớp

    moved = {"center": [10.4, 0.1, 0], "size": [1.9, 4.6, 1.6], "yaw": 0.25}
    r = await client.post(base + "/actions", json={"action": "EDIT_BOX", "object_id": "1", "box": moved})
    o1 = r.json()["objects"][0]
    assert o1["box"]["center"] == [10.4, 0.1, 0] and o1["original_box"]["center"] == [10, 0, 0]
    assert o1["review"]["action"] == "EDIT_BOX" and o1["review"]["final_label"] == "car"

    log = (await client.get("/api/v1/3d/corrections", params={"model": "pointpillars"})).json()
    assert [e["human_action"] for e in log][-2:] == ["ADD_BOX", "EDIT_BOX"]
    assert log[-1]["prediction"]["center"] == [10, 0, 0] and log[-1]["final_box"]["center"] == [10.4, 0.1, 0]
    assert log[-2]["prediction"] is None and log[-2]["source"] == "human"

    for oid in ("2", "3"):
        await client.post(base + "/actions", json={"action": "DELETE", "object_id": oid})
    assert (await client.post(base + "/approve", json={})).status_code == 200
    m = (await client.get("/api/v1/3d/metrics", params={"model": "pointpillars"})).json()
    assert m["added"] == 1 and m["edited"] == 1 and m["objects"] == 3
    await client.post("/api/v1/3d/export", params={"model": "pointpillars"})
    out = json.loads(next((store3d.root / "exports").glob("3d-*")).joinpath("labels3d_nusc.json").read_text())
    rows = out["results"]["tok-scene-0035_000"]
    assert {b["detection_name"] for b in rows} == {"car", "pedestrian"}
    assert next(b for b in rows if b["detection_name"] == "pedestrian")["source"] == "human"


@pytest.mark.asyncio
async def test_bev_image(client, store3d, dataroot):
    from PIL import Image

    # ảnh camera: nửa dưới (mặt đường trước xe) màu đỏ
    img = np.zeros((900, 1600, 3), np.uint8)
    img[450:] = [200, 30, 30]
    (dataroot / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(dataroot / "samples" / "CAM_FRONT" / "x.jpg")
    r = await client.get("/api/v1/3d/frames/pointpillars/scene-0035_000/bev", params={"range": 20, "res": 0.5})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert float(r.headers["x-ground-z"]) == pytest.approx(-1.84)  # point cloud giả toàn số 0 -> giá trị mặc định
    bev = np.asarray(Image.open(__import__("io").BytesIO(r.content)))
    assert bev.shape == (80, 80, 4)
    # camera trong test nhìn theo trục x: phía x > 0 có ảnh (đỏ), phía sau xe trống
    assert bev[40, 60, 3] == 255 and bev[40, 60, 0] > 150
    assert bev[40, 10, 3] == 0


@pytest.mark.asyncio
async def test_propagate_3d_api(client, store3d):
    nxt = frame3d("scene-0035_001")
    nxt.index = 1
    store3d.save_frame3d(nxt)
    base = "/api/v1/3d/frames/pointpillars/scene-0035_000"
    assert (await client.post(f"{base}/propagate", json={})).status_code == 409  # chưa approve
    await client.post(f"{base}/actions", json={"action": "CHANGE_CLASS", "object_id": "1", "label": "truck"})
    for oid in ("2", "3"):
        await client.post(f"{base}/actions", json={"action": "DELETE", "object_id": oid})
    assert (await client.post(f"{base}/approve", json={})).status_code == 200
    r = (await client.post(f"{base}/propagate", json={})).json()
    assert r["frames_updated"] == ["scene-0035_001"] and r["objects_propagated"] == 1 and r["objects_suppressed"] == 2
    f = (await client.get("/api/v1/3d/frames/pointpillars/scene-0035_001")).json()
    o = {x["object_id"]: x for x in f["objects"]}
    assert o["1"]["source"] == "propagated" and o["1"]["label"] == "truck" and o["1"]["propagation"]["distance_m"] == 0
    assert o["2"]["review"]["action"] == "PROPAGATED_DELETE"


@pytest.mark.asyncio
async def test_models_lists_project_models_not_built_yet(client, store3d):
    """Mô hình đơn lẻ có dự đoán trong work3d của dự án hiện trong danh sách với built = false; bản đã có frame đứng trước."""
    raw = store3d.root.parent / "work3d" / "preds" / "ssn" / "pred_instances_3d"
    raw.mkdir(parents=True)
    (raw / "results_nusc.json").write_text('{"results": {}}', encoding="utf-8")
    try:
        rows = (await client.get("/api/v1/3d/models")).json()
        assert [(r["model"], r["built"]) for r in rows] == [("pointpillars", True), ("ssn", False)]
        assert (await client.post("/api/v1/3d/models/nope/build", json={})).status_code == 404
        assert (await client.get("/api/v1/3d/models/build")).status_code == 200
    finally:
        import shutil

        shutil.rmtree(store3d.root.parent / "work3d")
