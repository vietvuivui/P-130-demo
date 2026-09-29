"""KITTI -> nuScenes.

Nhận hai kiểu thư mục (tìm ở bất kỳ độ sâu nào trong thư mục đã giải nén):
- object:   image_2/000123.png, velodyne/000123.bin, calib/000123.txt       (mỗi frame độc lập = một scene)
- tracking: image_02/0001/000123.png, velodyne/0001/000123.bin, calib/0001.txt, oxts/0001.txt (tuỳ chọn)
            (mỗi sequence = một scene, 10 Hz; có oxts thì tính tư thế xe, không có thì coi xe đứng yên)

Chỉ dùng camera trái màu (P2) làm CAM_FRONT. LiDAR KITTI: x trước, y trái, z lên, đặt cao ~1.73 m.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.services.ingest.nusc_writer import CameraInput, FrameInput, NuScenesWriter, read_points

VELO_HEIGHT = 1.73
IMG_EXT = (".png", ".jpg", ".jpeg")


def parse_calib(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            key, *vals = line.split()  # calib của tracking đôi khi không có dấu ':'
        else:
            key, rest = line.split(":", 1)
            vals = rest.split()
        try:
            out[key.strip()] = np.array([float(v) for v in vals])
        except ValueError:
            continue
    return out


def camera_from_calib(c: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(K 3x3, cam_from_velo 4x4) của camera 2 đã rectify."""
    p2 = c["P2"].reshape(3, 4)
    k = p2[:, :3]
    r0 = np.eye(4)
    r0[:3, :3] = c.get("R0_rect", c.get("R_rect", np.eye(3).ravel())).reshape(3, 3)
    tr = np.eye(4)
    tr[:3, :] = c.get("Tr_velo_to_cam", c.get("Tr_velo_cam")).reshape(3, 4)
    t2 = np.eye(4)
    t2[:3, 3] = np.linalg.solve(k, p2[:, 3])  # dịch của camera 2 so với camera 0 đã rectify
    return k, t2 @ r0 @ tr


def ego_from_velo() -> np.ndarray:
    m = np.eye(4)
    m[2, 3] = VELO_HEIGHT
    return m


def oxts_poses(path: Path) -> list[np.ndarray]:
    """Tư thế IMU theo từng dòng oxts (công thức của KITTI devkit: Mercator theo vĩ độ dòng đầu)."""
    rows = [np.array([float(v) for v in ln.split()]) for ln in Path(path).read_text().splitlines() if ln.strip()]
    er = 6378137.0
    scale = math.cos(rows[0][0] * math.pi / 180.0)
    poses, origin = [], None
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        tx = scale * lon * math.pi * er / 180
        ty = scale * er * math.log(math.tan((90 + lat) * math.pi / 360))
        rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        m = np.eye(4)
        m[:3, :3] = rz @ ry @ rx
        m[:3, 3] = [tx, ty, alt]
        if origin is None:
            origin = np.linalg.inv(m)
        poses.append(origin @ m)
    return poses


def _find_dirs(root: Path, name: str) -> list[Path]:
    return sorted(p for p in root.rglob(name) if p.is_dir())


def detect(root: Path) -> str | None:
    if _find_dirs(root, "velodyne") and (_find_dirs(root, "image_2") or _find_dirs(root, "image_02")):
        return "kitti"
    return None


def convert(root: Path, out: Path, version: str = "v1.0-custom", max_frames: int | None = None, progress=None) -> dict:
    writer = NuScenesWriter(out, version, location="kitti")
    n_frames = 0
    scenes = []
    for velo_dir in _find_dirs(root, "velodyne"):
        base = velo_dir.parent
        subdirs = sorted(d for d in velo_dir.iterdir() if d.is_dir())
        if subdirs:  # tracking: velodyne/<seq>/*.bin
            for seq in subdirs:
                calib = base / "calib" / f"{seq.name}.txt"
                img_dir = base / "image_02" / seq.name
                if not calib.exists() or not img_dir.exists():
                    continue
                k, cam_from_velo = camera_from_calib(parse_calib(calib))
                oxts = base / "oxts" / f"{seq.name}.txt"
                poses = oxts_poses(oxts) if oxts.exists() else None
                frames = []
                for i, bin_path in enumerate(sorted(seq.glob("*.bin"))):
                    img = next(
                        (
                            img_dir / f"{bin_path.stem}{e}"
                            for e in IMG_EXT
                            if (img_dir / f"{bin_path.stem}{e}").exists()
                        ),
                        None,
                    )
                    if img is None:
                        continue
                    idx = int(bin_path.stem) if bin_path.stem.isdigit() else i
                    frames.append(FrameInput(
                        timestamp_s=idx * 0.1, points=read_points(bin_path), ego_from_lidar=ego_from_velo(),
                        cameras={"CAM_FRONT": CameraInput(img, k, cam_from_velo)},
                        global_from_ego=poses[idx] if poses and idx < len(poses) else None, name=bin_path.stem,
                    ))  # fmt: skip
                    if max_frames and n_frames + len(frames) >= max_frames:
                        break
                if frames:
                    scenes.append(writer.add_scene(f"kitti-{seq.name}", frames, "KITTI tracking"))
                    n_frames += len(frames)
                    if progress:
                        progress(n_frames)
                if max_frames and n_frames >= max_frames:
                    break
        else:  # object: mỗi frame một scene
            img_dir = base / "image_2"
            for bin_path in sorted(velo_dir.glob("*.bin")):
                calib = base / "calib" / f"{bin_path.stem}.txt"
                img = next(
                    (img_dir / f"{bin_path.stem}{e}" for e in IMG_EXT if (img_dir / f"{bin_path.stem}{e}").exists()),
                    None,
                )
                if img is None or not calib.exists():
                    continue
                k, cam_from_velo = camera_from_calib(parse_calib(calib))
                f = FrameInput(timestamp_s=0.0, points=read_points(bin_path), ego_from_lidar=ego_from_velo(),
                               cameras={"CAM_FRONT": CameraInput(img, k, cam_from_velo)}, name=bin_path.stem)  # fmt: skip
                scenes.append(writer.add_scene(f"kitti-{bin_path.stem}", [f], "KITTI object"))
                n_frames += 1
                if progress and n_frames % 10 == 0:
                    progress(n_frames)
                if max_frames and n_frames >= max_frames:
                    break
        if max_frames and n_frames >= max_frames:
            break
    if not n_frames:
        raise ValueError("Không tìm thấy frame KITTI nào (cần velodyne/*.bin + image_2/*.png + calib/*.txt)")
    writer.write()
    return dict(version=version, scenes=len(scenes), frames=n_frames, has_lidar=True, cameras=["CAM_FRONT"])
