"""Nhận dữ liệu của end-user: KITTI và "LiDAR rời + camera" đổi sang nuScenes, giải nén an toàn, nhận dạng loại."""

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from src.services import ingest
from src.services.ingest import kitti, lidarcam
from src.services.ingest.nusc_writer import read_points
from src.services.nuscenes_data import NuScenesMini

P2 = np.array([[721.5, 0, 609.6, 44.9], [0, 721.5, 172.9, 0.2], [0, 0, 1, 0.003]])
R0 = np.eye(3)
# velo (x trước, y trái, z lên) -> camera (x phải, y xuống, z trước), camera lệch 0.27 m về trước, thấp hơn 0.08 m
TR = np.array([[0.0, -1, 0, 0], [0, 0, -1, -0.08], [1, 0, 0, -0.27]])


def write_kitti(root, n=2):
    for d in ("velodyne", "image_2", "calib"):
        (root / "training" / d).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        pts = np.array([[10.0, 0.0, 0.0, 0.5], [20.0, 2.0, -1.0, 0.2], [5.0, -1.0, 0.5, 1.0]], np.float32)
        pts.tofile(root / "training" / "velodyne" / f"{i:06d}.bin")
        Image.new("RGB", (1242, 375), (90, 90, 90)).save(root / "training" / "image_2" / f"{i:06d}.png")
        lines = [f"P{k}: " + " ".join(map(str, P2.ravel())) for k in range(4)]
        lines += ["R0_rect: " + " ".join(map(str, R0.ravel())), "Tr_velo_to_cam: " + " ".join(map(str, TR.ravel()))]
        (root / "training" / "calib" / f"{i:06d}.txt").write_text("\n".join(lines))


def test_kitti_to_nuscenes_geometry(tmp_path):
    write_kitti(tmp_path / "src")
    assert ingest.detect_kind(tmp_path / "src") == "kitti"
    info = kitti.convert(tmp_path / "src", tmp_path / "ds")
    assert info["frames"] == 2 and info["scenes"] == 2  # KITTI object: mỗi frame một scene
    data = NuScenesMini(tmp_path / "ds", "v1.0-custom")
    scene, idx, tok = data.keyframes()[0]
    kf = data._keyframe_data[tok]
    lidar = data.sample_data[kf["LIDAR_TOP"]]
    pts = np.fromfile(tmp_path / "ds" / lidar["filename"], np.float32).reshape(-1, 5)
    # điểm KITTI (10, 0, 0) phía trước -> hệ LiDAR nuScenes (0, 10, 0); cường độ 0-1 -> 0-255
    assert pts[0, :3] == pytest.approx([0, 10, 0]) and pts[0, 3] == pytest.approx(127.5)
    # chiếu điểm bằng bảng nuScenes phải trùng chiếu bằng calib KITTI gốc
    frame = data.camera_frame(tok, "CAM_FRONT", [], scene, idx)
    uv_nusc, depth = data.lidar_in_image(frame, min_depth=0.5)
    velo = np.array([[10.0, 0, 0, 1], [20.0, 2, -1, 1], [5.0, -1, 0.5, 1]])
    cam = P2 @ np.vstack([np.hstack([R0 @ TR[:, :3], (R0 @ TR[:, 3:])]), [0, 0, 0, 1]]) @ velo.T
    uv_kitti = (cam[:2] / cam[2]).T
    assert uv_nusc == pytest.approx(uv_kitti[: len(uv_nusc)], abs=0.05)


def test_lidarcam_with_poses_and_sweeps(tmp_path):
    root = tmp_path / "data"
    (root / "lidar").mkdir(parents=True)
    (root / "front").mkdir()
    cam_from_lidar = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    calib = {"frame_rate": 10, "lidar": {"axes": "x_forward"},
             "cameras": {"front": {"intrinsic": [[1000, 0, 800], [0, 1000, 450], [0, 0, 1]], "lidar_to_camera": cam_from_lidar}}}  # fmt: skip
    (root / "calib.json").write_text(json.dumps(calib))
    poses = {}
    for i in range(3):
        np.array([[10.0, 0, 0, 0.3]], np.float32).tofile(root / "lidar" / f"{i:04d}.bin")
        Image.new("RGB", (1600, 900)).save(root / "front" / f"{i:04d}.jpg")
        m = np.eye(4)
        m[0, 3] = 2.0 * i  # xe chạy thẳng 2 m mỗi frame
        poses[f"{i:04d}"] = m.tolist()
    (root / "poses.json").write_text(json.dumps(poses))
    z = tmp_path / "up.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in root.rglob("*"):
            zf.write(f, f.relative_to(tmp_path))
    ext = ingest.safe_extract(z, tmp_path / "ext")
    assert ingest.detect_kind(ext) == "lidar_cam"
    info = lidarcam.convert(ext, tmp_path / "ds")
    assert info["frames"] == 3 and info["cameras"] == ["CAM_FRONT"]  # camera "front" thành CAM_FRONT
    data = NuScenesMini(tmp_path / "ds", "v1.0-custom")
    keys = data.keyframes()
    assert [k[1] for k in keys] == [0, 1, 2]
    last = data._keyframe_data[keys[2][2]]
    sd = data.sample_data[last["LIDAR_TOP"]]
    assert sd["prev"]  # frame trước làm sweep
    pose = data.ego_pose[sd["ego_pose_token"]]
    assert pose["translation"][0] == pytest.approx(4.0)
    # LiDAR mặc định cao 1.8 m: điểm (10, 0, 0) của LiDAR ở độ cao 1.8 m trong hệ ego
    g = data._global_from_ego(last["LIDAR_TOP"]) @ data._ego_from_sensor(last["LIDAR_TOP"])
    p = np.fromfile(tmp_path / "ds" / sd["filename"], np.float32).reshape(-1, 5)[0]
    world = g @ np.array([*p[:3], 1.0])
    assert world[:3] == pytest.approx([14.0, 0.0, 1.8])


def test_safe_extract_blocks_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../outside.txt", "x")
    with pytest.raises(ingest.IngestError):
        ingest.safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


def test_read_pcd_ascii_and_binary(tmp_path):
    head = (
        "VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\n"
    )
    (tmp_path / "a.pcd").write_text(head + "DATA ascii\n1 2 3 4\n5 6 7 8\n")
    assert read_points(tmp_path / "a.pcd").tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    body = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32).tobytes()
    (tmp_path / "b.pcd").write_bytes((head + "DATA binary\n").encode() + body)
    assert read_points(tmp_path / "b.pcd").tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_detect_kinds(tmp_path):
    (tmp_path / "imgs").mkdir()
    Image.new("RGB", (64, 48)).save(tmp_path / "imgs" / "a.jpg")
    assert ingest.detect_kind(tmp_path / "imgs") == "images"
    (tmp_path / "nu" / "v1.0-mini").mkdir(parents=True)
    for t in ("scene", "sample_data"):
        (tmp_path / "nu" / "v1.0-mini" / f"{t}.json").write_text("[]")
    assert ingest.detect_kind(tmp_path / "nu") == "nuscenes"
    assert ingest.find_nuscenes(tmp_path / "nu") == (tmp_path / "nu", "v1.0-mini")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "x.txt").write_text("?")
    with pytest.raises(ingest.IngestError):
        ingest.detect_kind(tmp_path / "junk")
