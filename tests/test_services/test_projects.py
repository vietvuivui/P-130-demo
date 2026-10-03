"""Dự án của end-user: tải dữ liệu lên -> xử lý nền -> duyệt qua API theo dự án -> xuất nuScenes."""

import json
import zipfile

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.main import app
from src.models.schemas3d import Box3D, Object3D, Verify3D
from src.services import projects
from src.services.export_nusc import export_nuscenes
from src.services.projects import MODEL3D, ProjectManager, ProjectOptions
from tests.test_services.test_ingest import write_kitti

DEMO_RED = (214, 38, 38)  # màu "car" của detector demo


def demo_image(path, with_car=True):
    img = Image.new("RGB", (640, 360), (60, 60, 60))
    if with_car:
        img.paste(DEMO_RED, (200, 150, 320, 230))
    img.save(path)


@pytest.fixture
def manager(tmp_path, config, monkeypatch):
    cfg = config.model_copy(deep=True)
    cfg.detection.detectors = ["demo"]
    m = ProjectManager(tmp_path / "projects", lambda: cfg, mm3d_python=str(tmp_path / "no-mm3d"))
    monkeypatch.setattr(projects, "_manager", m)
    return m


def test_images_project_runs_2d(manager, tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"img{i}.jpg"
        demo_image(f, with_car=i != 1)
        files.append(f)
    p = manager.create("Ảnh thử", files, options=ProjectOptions(sequential=False))
    p = manager.run(p.id)
    assert p.status == "ready" and p.kind == "images"
    assert [s.status for s in p.steps] == ["done", "done"]
    frames = manager.store(p.id).list_frames()
    assert len(frames) == 3  # ảnh rời: mỗi ảnh một keyframe
    assert sum(len(f.objects) for f in frames) == 2 and frames[0].objects[0].label == "car"


def test_kitti_project_without_3d_env_keeps_2d(manager, tmp_path):
    write_kitti(tmp_path / "k")
    for f in (tmp_path / "k" / "training" / "image_2").glob("*.png"):
        img = Image.open(f)
        img.paste(DEMO_RED, (500, 150, 640, 250))
        img.save(f)
    z = tmp_path / "kitti.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "k").rglob("*"):
            zf.write(f, f.relative_to(tmp_path / "k"))
    p = manager.run(manager.create("KITTI", [z]).id)
    assert p.kind == "kitti" and p.status == "ready", p.message
    st = {s.name: s.status for s in p.steps}
    assert st == {"ingest": "done", "label2d": "done", "predict3d": "skipped", "fuse2d": "skipped",
                  "verify3d": "skipped"}  # fmt: skip
    assert "môi trường 3D" in next(s.message for s in p.steps if s.name == "predict3d")
    assert len(manager.store(p.id).list_frames()) == 2
    root, version = manager.dataset(p.id)
    assert (root / version / "sample.json").exists()


def test_export_nuscenes_links_tracks(manager, tmp_path):
    write_kitti(tmp_path / "k")
    placeholder = tmp_path / "placeholder.zip"
    placeholder.write_bytes(b"")
    p = manager.create("KITTI", [placeholder])
    root = manager.dir(p.id) / "dataset"
    from src.services.ingest import kitti

    kitti.convert(tmp_path / "k", root)
    p.stats.update(version="v1.0-custom", has_lidar=True)
    manager.save(p)
    store = manager.store(p.id)
    from src.services.nuscenes_data import NuScenesMini

    data = NuScenesMini(root, "v1.0-custom")
    from src.models.schemas3d import Frame3DRecord

    for k, (scene, idx, tok) in enumerate(data.keyframes()):
        objs = [
            Object3D(object_id="1", label="car", score=0.8, track_id="0-0",
                     box=Box3D(center=[1.0 + k, 10, -1], size=[1.9, 4.5, 1.6], yaw=1.57),
                     verify=Verify3D(verdict="DUNG", level="low", comment="", lidar_points=30)),
            Object3D(object_id="2", label="pedestrian", score=0.5, box=Box3D(center=[3, 5, -1], size=[0.6, 0.7, 1.7], yaw=0.0)),
        ]  # fmt: skip
        objs[0].review.status, objs[0].review.final_label = "approved", "car"
        objs[1].review.status = "deleted" if k == 0 else "approved"
        f = Frame3DRecord(frame_id=f"{scene}_{idx:03d}", model=MODEL3D, sample_token=tok, scene="kitti", index=k,
                          lidar_sd_token="x", global_from_lidar=np.eye(4).tolist(), objects=objs, status="approved")  # fmt: skip
        store.save_frame3d(f)
    info = export_nuscenes(store, root, "v1.0-custom", MODEL3D, tmp_path / "out.zip")
    assert info["annotations"] == 3 and info["instances"] == 2  # car cùng track = 1 instance, 1 pedestrian
    with zipfile.ZipFile(tmp_path / "out.zip") as z:
        anns = json.loads(z.read("v1.0-custom/sample_annotation.json"))
        inst = json.loads(z.read("v1.0-custom/instance.json"))
        cats = {c["token"]: c["name"] for c in json.loads(z.read("v1.0-custom/category.json"))}
        assert "labels3d_nusc.json" in z.namelist()
    car = next(i for i in inst if cats[i["category_token"]] == "vehicle.car")
    assert car["nbr_annotations"] == 2
    first = next(a for a in anns if a["token"] == car["first_annotation_token"])
    assert first["next"] and first["num_lidar_pts"] == 30


@pytest_asyncio.fixture
async def api(manager):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_project_api_upload_review_export(api, manager, tmp_path):
    files = []
    for i in range(2):
        f = tmp_path / f"a{i}.png"
        demo_image(f)
        files.append(("files", (f.name, f.read_bytes(), "image/png")))
    r = await api.post("/api/v1/projects", files=files, data={"name": "Bộ ảnh", "sequential": "false"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    manager.wait_idle(60)
    p = (await api.get(f"/api/v1/projects/{pid}")).json()
    assert p["status"] == "ready" and p["kind"] == "images"
    # đổi tên dự án trên UI: lưu vào project.json, id giữ nguyên; tên rỗng bị từ chối
    r = await api.patch(f"/api/v1/projects/{pid}", json={"name": "  Bộ ảnh   đường phố "})
    assert r.status_code == 200 and r.json()["name"] == "Bộ ảnh đường phố" and r.json()["id"] == pid
    assert (await api.get(f"/api/v1/projects/{pid}")).json()["name"] == "Bộ ảnh đường phố"
    assert (await api.patch(f"/api/v1/projects/{pid}", json={"name": ""})).status_code == 422
    assert (await api.patch("/api/v1/projects/p-khong-co", json={"name": "x"})).status_code == 404
    frames = (await api.get(f"/p/{pid}/api/v1/frames")).json()
    assert len(frames) == 2
    fid = frames[0]["frame_id"]
    img = await api.get(f"/p/{pid}/api/v1/frames/{fid}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    frame = (await api.get(f"/p/{pid}/api/v1/frames/{fid}")).json()
    for o in frame["objects"]:
        await api.post(f"/p/{pid}/api/v1/frames/{fid}/actions", json={"action": "KEEP", "object_id": o["object_id"]})
    assert (await api.post(f"/p/{pid}/api/v1/frames/{fid}/approve", json={})).status_code == 200
    assert (await api.post(f"/api/v1/projects/{pid}/export", params={"format": "nuscenes"})).status_code == 409
    r = await api.post(f"/api/v1/projects/{pid}/export", params={"format": "coco"})
    assert r.status_code == 200, r.text
    dl = await api.get(r.json()["url"])
    assert dl.status_code == 200 and dl.headers["content-type"] == "application/zip"
    # dữ liệu không nhận ra -> dự án lỗi, có thông báo
    bad = await api.post("/api/v1/projects", files=[("files", ("x.zip", _zip_bytes(tmp_path), "application/zip"))])
    manager.wait_idle(60)
    pb = (await api.get(f"/api/v1/projects/{bad.json()['id']}")).json()
    assert pb["status"] == "error" and "Không nhận ra" in pb["message"]
    assert (await api.delete(f"/api/v1/projects/{pid}")).status_code == 200
    assert (await api.get(f"/api/v1/projects/{pid}")).status_code == 404


def _zip_bytes(tmp_path):
    z = tmp_path / "junk.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("notes.txt", "không phải dữ liệu")
    return z.read_bytes()


def test_models3d_names_are_validated(tmp_path, config):
    with pytest.raises(ValueError):
        ProjectManager(tmp_path / "p", lambda: config, models3d=["centerpoint_voxel", "x; rm -rf /"])
    assert ProjectManager(tmp_path / "q", lambda: config, models3d=["centerpoint_pillar"]).models3d == [
        "centerpoint_pillar"
    ]


def test_fuse2d_adds_projected_3d_boxes_to_auto_frames(manager, tmp_path):
    write_kitti(tmp_path / "k")
    z = tmp_path / "kitti.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "k").rglob("*"):
            zf.write(f, f.relative_to(tmp_path / "k"))
    p = manager.run(manager.create("KITTI", [z]).id)
    store = manager.store(p.id)
    from src.services.nuscenes_data import NuScenesMini

    root, version = manager.dataset(p.id)
    data = NuScenesMini(root, version)
    keys = data.keyframes()
    first = store.load_frame(f"{keys[0][0]}_{keys[0][1]:03d}")
    first.status = "in_review"  # người đã bắt đầu duyệt: không được ghi đè
    store.save_frame(first)
    preds = {}
    for scene, idx, tok in keys:
        cam_sd = data.camera_frame(tok, "CAM_FRONT", [], scene, idx).image.sd_token
        center = np.linalg.inv(data.cam_from_global(cam_sd)) @ np.array([0.0, 0.0, 15.0, 1.0])  # 15 m trước camera
        preds[tok] = [{"sample_token": tok, "translation": center[:3].tolist(), "size": [1.0, 1.0, 1.0],
                       "rotation": [1, 0, 0, 0], "detection_name": "pedestrian", "detection_score": 0.7}]  # fmt: skip
    work = manager.dir(p.id) / "work3d"
    work.mkdir(exist_ok=True)
    (work / "preds.json").write_text(json.dumps({"results": preds}))
    p = manager.get(p.id)
    next(s for s in p.steps if s.name == "fuse2d").status = "pending"
    manager.save(p)
    p = manager.run(p.id)
    assert next(s for s in p.steps if s.name == "fuse2d").status == "done" and p.stats["fused2d"] == len(keys) - 1
    frames = {f.frame_id: f for f in store.list_frames()}
    kept = frames[first.frame_id]
    assert kept.status == "in_review" and not any("lidar3d" in o.models for o in kept.objects)
    for fid, f in frames.items():
        if fid != first.frame_id:
            assert "lidar3d" in f.detectors
            assert any(o.label == "pedestrian" and o.models.get("lidar3d") == 0.7 for o in f.objects)
