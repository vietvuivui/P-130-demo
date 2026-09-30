"""Định dạng "LiDAR rời + camera" của AutoLabel -> nuScenes.

Cấu trúc (zip hoặc thư mục; tên frame giống nhau giữa lidar và camera, sắp theo tên = thứ tự thời gian):

    calib.json
    lidar/000000.bin | .pcd | .npy        float32 x y z [cường độ]
    CAM_FRONT/000000.jpg | .png           mỗi camera một thư mục, tên thư mục = tên camera trong calib.json
    poses.json            (tuỳ chọn)      {"000000": [[4x4 ego -> thế giới]], ...}
    timestamps.json       (tuỳ chọn)      {"000000": 12.30, ...}  giây; không có thì dùng frame_rate

calib.json:
    {
      "frame_rate": 10,                                  (tuỳ chọn, mặc định 10 Hz)
      "lidar": {"axes": "x_forward", "to_ego": [[4x4]]},   axes: "x_forward" (x trước, y trái) hoặc "x_right" (nuScenes);
                                                           to_ego mặc định: LiDAR cao 1.8 m, cùng hướng xe
      "cameras": {
        "CAM_FRONT": {"intrinsic": [[fx,0,cx],[0,fy,cy],[0,0,1]], "lidar_to_camera": [[4x4]]}
      }
    }

Camera tên khác bộ nuScenes (CAM_FRONT, CAM_FRONT_LEFT, ...) được gán lần lượt vào tên nuScenes còn trống; camera đầu
tiên luôn là CAM_FRONT (camera dùng cho gán nhãn 2D).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.services.ingest.nusc_writer import NUSC_CAMERAS, CameraInput, FrameInput, NuScenesWriter, read_points

LIDAR_EXT = (".bin", ".pcd", ".npy")
IMG_EXT = (".jpg", ".jpeg", ".png")


def find_root(root: Path) -> Path | None:
    for calib in sorted(Path(root).rglob("calib.json")):
        if (calib.parent / "lidar").is_dir():
            return calib.parent
    return None


def detect(root: Path) -> str | None:
    return "lidar_cam" if find_root(root) else None


def _mat(v, name: str) -> np.ndarray:
    m = np.asarray(v, float)
    if m.shape != (4, 4):
        raise ValueError(f"calib.json: {name} phải là ma trận 4x4")
    return m


def camera_names(names: list[str]) -> dict[str, str]:
    """Tên camera của người dùng -> kênh nuScenes. Không có camera nào tên CAM_FRONT thì camera đầu tiên là CAM_FRONT."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    if "CAM_FRONT" not in names:
        out[names[0]] = "CAM_FRONT"
        taken.add("CAM_FRONT")
    for n in names:
        if n not in out and n in NUSC_CAMERAS and n not in taken:
            out[n] = n
            taken.add(n)
    free = [c for c in NUSC_CAMERAS if c not in taken]
    for n in names:
        if n not in out:
            if not free:
                raise ValueError("Tối đa 6 camera")
            out[n] = free.pop(0)
    return out


def convert(src: Path, out: Path, version: str = "v1.0-custom", max_frames: int | None = None, progress=None) -> dict:
    root = find_root(src)
    if root is None:
        raise ValueError("Không thấy calib.json cạnh thư mục lidar/")
    calib = json.loads((root / "calib.json").read_text(encoding="utf-8"))
    lid = calib.get("lidar", {})
    axes = lid.get("axes", "x_forward")
    if axes not in ("x_forward", "x_right"):
        raise ValueError("calib.json: lidar.axes phải là x_forward hoặc x_right")
    ego_from_lidar = _mat(lid["to_ego"], "lidar.to_ego") if "to_ego" in lid else np.diag([1.0, 1, 1, 1])
    if "to_ego" not in lid:
        ego_from_lidar[2, 3] = 1.8
    cams_cfg = calib.get("cameras", {})
    if not cams_cfg:
        raise ValueError("calib.json: cần ít nhất một camera trong 'cameras'")
    mapping = camera_names(list(cams_cfg))
    cams = {n: (np.asarray(c["intrinsic"], float), _mat(c["lidar_to_camera"], f"cameras.{n}.lidar_to_camera"))
            for n, c in cams_cfg.items()}  # fmt: skip
    poses = json.loads((root / "poses.json").read_text()) if (root / "poses.json").exists() else {}
    stamps = json.loads((root / "timestamps.json").read_text()) if (root / "timestamps.json").exists() else {}
    rate = float(calib.get("frame_rate", 10))
    lidar_files = sorted(p for p in (root / "lidar").iterdir() if p.suffix.lower() in LIDAR_EXT)
    if max_frames:
        lidar_files = lidar_files[:max_frames]
    frames = []
    for i, lf in enumerate(lidar_files):
        cameras = {}
        for n, (k, cam_from_lidar) in cams.items():
            img = next((root / n / f"{lf.stem}{e}" for e in IMG_EXT if (root / n / f"{lf.stem}{e}").exists()), None)
            if img is not None:
                cameras[mapping[n]] = CameraInput(img, k, cam_from_lidar)
        if "CAM_FRONT" not in cameras:
            continue  # gán nhãn 2D cần ảnh camera chính ở mọi frame
        pose = poses.get(lf.stem)
        frames.append(FrameInput(
            timestamp_s=float(stamps.get(lf.stem, i / rate)), points=read_points(lf), ego_from_lidar=ego_from_lidar,
            lidar_axes=axes, cameras=cameras, global_from_ego=None if pose is None else _mat(pose, f"poses[{lf.stem}]"),
            name=lf.stem,
        ))  # fmt: skip
        if progress and len(frames) % 10 == 0:
            progress(len(frames))
    if not frames:
        raise ValueError("Không có frame nào đủ LiDAR + ảnh camera chính cùng tên")
    writer = NuScenesWriter(out, version, location="custom")
    writer.add_scene("seq-0001", frames, "LiDAR + camera")
    writer.write()
    return dict(version=version, scenes=1, frames=len(frames), has_lidar=True, cameras=sorted(set(mapping.values())),
                has_poses=bool(poses))  # fmt: skip
