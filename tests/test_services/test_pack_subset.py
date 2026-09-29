"""scripts/pack_nuscenes_subset.py: gói vài scene + workspace, giải nén ra loader của repo đọc được như bản đầy đủ."""

import importlib.util
import json
import zipfile

import pytest

from src.services.nuscenes_data import NuScenesMini
from tests.conftest import ROOT

spec = importlib.util.spec_from_file_location("pack_subset", ROOT / "scripts" / "pack_nuscenes_subset.py")
pack_subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack_subset)


def fake_nuscenes(root):
    """2 scene; mỗi scene 2 keyframe, giữa là 1 sweep camera; LiDAR keyframe + sweep; chuỗi next nối sang scene sau."""
    t = root / "v1.0-mini"
    t.mkdir(parents=True)
    tables = {
        "scene": [],
        "sample": [],
        "sample_data": [],
        "ego_pose": [],
        "sample_annotation": [],
        "instance": [],
        "sensor": [
            {"token": "s-cam", "channel": "CAM_FRONT"},
            {"token": "s-lid", "channel": "LIDAR_TOP"},
            {"token": "s-back", "channel": "CAM_BACK"},
        ],
        "calibrated_sensor": [
            {"token": "cs-cam", "sensor_token": "s-cam"},
            {"token": "cs-lid", "sensor_token": "s-lid"},
            {"token": "cs-back", "sensor_token": "s-back"},
        ],
        "category": [{"token": "c-car", "name": "vehicle.car"}],
    }
    chains = {"cs-cam": [], "cs-lid": [], "cs-back": []}
    ts = 0
    for sc in ("A", "B"):
        tables["scene"].append({"token": f"sc{sc}", "name": f"scene-{sc}", "first_sample_token": f"{sc}0"})
        tables["instance"].append({"token": f"i{sc}", "category_token": "c-car"})
        for k in range(2):
            tables["sample"].append(
                {"token": f"{sc}{k}", "scene_token": f"sc{sc}", "next": f"{sc}{k + 1}" if k == 0 else ""}
            )
            tables["sample_annotation"].append(
                {"token": f"a{sc}{k}", "sample_token": f"{sc}{k}", "instance_token": f"i{sc}"}
            )
            for cs, folder, sweeps in (
                ("cs-cam", "CAM_FRONT", 1),
                ("cs-lid", "LIDAR_TOP", 1),
                ("cs-back", "CAM_BACK", 0),
            ):
                for j in range(1 + (sweeps if k == 0 else 0)):
                    ts += 10
                    key = j == 0
                    tok = f"{cs}-{sc}{k}-{j}"
                    chains[cs].append(tok)
                    tables["sample_data"].append(
                        {
                            "token": tok,
                            "sample_token": f"{sc}{k}",
                            "calibrated_sensor_token": cs,
                            "ego_pose_token": f"ep-{tok}",
                            "is_key_frame": key,
                            "timestamp": ts,
                            "filename": f"{'samples' if key else 'sweeps'}/{folder}/{tok}.jpg",
                            "width": 1600,
                            "height": 900,
                        }
                    )
                    tables["ego_pose"].append({"token": f"ep-{tok}"})
    by_tok = {sd["token"]: sd for sd in tables["sample_data"]}
    for chain in chains.values():
        for prev, a, b in zip([""] + chain[:-1], chain, chain[1:] + [""], strict=True):
            by_tok[a]["next"] = b
            by_tok[a]["prev"] = prev
    for name, rows in tables.items():
        (t / f"{name}.json").write_text(json.dumps(rows))
    for sd in tables["sample_data"]:
        f = root / sd["filename"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 10)


def fake_workspace(ws):
    for sc in ("A", "B"):
        for sub in ("frames", "lidar", "gt"):
            (ws / sub).mkdir(parents=True, exist_ok=True)
            (ws / sub / f"scene-{sc}_000.json").write_text("{}")
        cache = ws / "cache" / "detections" / "yolo_world-abc"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"cs-cam-{sc}0-0.json").write_text("[]")
    (ws / "corrections.jsonl").write_text("")
    (ws / "exports" / "x").mkdir(parents=True)
    (ws / "exports" / "x" / "coco.json").write_text("{}")


def test_pack_one_scene_and_load_it_back(tmp_path):
    data, ws = tmp_path / "nuscenes", tmp_path / "ws"
    fake_nuscenes(data)
    fake_workspace(ws)
    info = pack_subset.pack(data, "v1.0-mini", ["scene-A"], tmp_path / "out.zip", ws)
    assert info["keyframes"] == 2 and info["images"] == 3 and info["workspace_frames"] == 1

    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "x")
    root = "autolabel_subset/"
    assert root + "workspace/frames/scene-A_000.json" in names
    assert root + "workspace/frames/scene-B_000.json" not in names
    assert root + "workspace/cache/detections/yolo_world-abc/cs-cam-A0-0.json" in names
    assert root + "workspace/cache/detections/yolo_world-abc/cs-cam-B0-0.json" not in names
    assert not any("exports" in n or "CAM_BACK" in n or "B0" in n for n in names)
    assert not any("LIDAR_TOP" in n and n.endswith(".jpg") for n in names)  # không --with-lidar thì không kèm file
    assert root + "README.txt" in names

    # Loader của repo đọc bảng đã lọc như bản đầy đủ; chuỗi next bị cắt ở ranh giới scene
    full = NuScenesMini(data).camera_timeline("scene-A", "CAM_FRONT")
    sub = NuScenesMini(tmp_path / "x" / root / "nuscenes")
    assert sub.camera_timeline("scene-A", "CAM_FRONT") == full
    assert [k[0] for k in sub.keyframes()] == ["scene-A", "scene-A"]
    assert all((tmp_path / "x" / root / "nuscenes" / im.path).is_file() for im in full)
    assert len(sub.sample_annotation) == 2 and set(sub.category_of_instance) == {"iA"}


def test_pack_with_lidar_and_unknown_scene(tmp_path):
    data = tmp_path / "nuscenes"
    fake_nuscenes(data)
    info = pack_subset.pack(data, "v1.0-mini", ["scene-B"], tmp_path / "b.zip", with_lidar=True)
    assert info["lidar"] == 2  # chỉ LiDAR keyframe
    with pytest.raises(SystemExit, match="scene-Z"):
        pack_subset.pack(data, "v1.0-mini", ["scene-Z"], tmp_path / "z.zip")


def test_random_scenes_only_where_images_exist_and_window(tmp_path):
    data = tmp_path / "nuscenes"
    fake_nuscenes(data)
    for f in (data / "samples" / "CAM_FRONT").glob("*B0*"):
        f.unlink()  # như bản trainval mới tải vài phần blob: scene B không có ảnh trên máy
    assert pack_subset.pick_random_scenes(data, "v1.0-mini", 2, seed=1) == ["scene-A"]

    info = pack_subset.pack(data, "v1.0-mini", ["scene-A"], tmp_path / "w.zip", window=1, seed=3)
    assert info["keyframes"] == 1
    with zipfile.ZipFile(tmp_path / "w.zip") as zf:
        zf.extractall(tmp_path / "x")
    sub = NuScenesMini(tmp_path / "x" / "autolabel_subset" / "nuscenes")
    assert len(sub.keyframes()) == 1 and sub.camera_timeline("scene-A", "CAM_FRONT")


def test_pack_for_3d_all_cameras_lidar_sweeps_and_split(tmp_path):
    data = tmp_path / "nuscenes"
    fake_nuscenes(data)
    info = pack_subset.pack(data, "v1.0-mini", ["scene-A"], tmp_path / "3d.zip", all_cameras=True, lidar_sweeps=1)
    assert info["images"] == 5  # CAM_FRONT 2 keyframe + 1 sweep, CAM_BACK 2 keyframe
    assert info["lidar"] == 3  # 2 LiDAR keyframe + 1 lần quét liền trước keyframe thứ hai
    with zipfile.ZipFile(tmp_path / "3d.zip") as zf:
        names = zf.namelist()
    assert sum("sweeps/LIDAR_TOP" in n for n in names) == 1
    assert sum("samples/CAM_BACK" in n for n in names) == 2
    assert pack_subset.pick_random_scenes(data, "v1.0-mini", 5, seed=0, allowed={"scene-B"}) == ["scene-B"]
