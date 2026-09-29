"""Nhận dữ liệu của end-user: giải nén an toàn, nhận dạng loại dữ liệu, chuyển về dạng pipeline dùng được.

Loại dữ liệu:
- video      mp4/mov/avi/mkv/webm                       -> chế độ Video (chỉ 2D)
- images     ảnh jpg/png rời hoặc zip ảnh                -> chuỗi ảnh (chỉ 2D)
- nuscenes   zip theo cấu trúc nuScenes (v1.0-*/scene.json + samples/ + sweeps/)   -> 2D + 3D
- kitti      zip KITTI object / tracking (velodyne + image_2|image_02 + calib)     -> đổi sang nuScenes, 2D + 3D
- lidar_cam  zip "LiDAR rời + camera" (calib.json + lidar/ + <camera>/)            -> đổi sang nuScenes, 2D + 3D
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from src.services.ingest import kitti, lidarcam

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz")
KINDS = ("video", "images", "nuscenes", "kitti", "lidar_cam")


class IngestError(ValueError):
    pass


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_EXT)


def safe_extract(archive: Path, dest: Path) -> Path:
    """Giải nén, chặn đường dẫn thoát ra ngoài thư mục đích (zip-slip) và link."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (root / info.filename).resolve()
                if not target.is_relative_to(root):
                    raise IngestError(f"File trong zip có đường dẫn không hợp lệ: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                target = (root / m.name).resolve()
                if not target.is_relative_to(root) or m.issym() or m.islnk():
                    raise IngestError(f"File trong tar không hợp lệ: {m.name}")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
    else:
        raise IngestError(f"Không giải nén được {archive.name} (hỗ trợ .zip, .tar, .tar.gz)")
    return dest


def find_nuscenes(root: Path) -> tuple[Path, str] | None:
    """(dataroot, version) nếu trong thư mục có bảng nuScenes (v1.0-*/scene.json)."""
    for scene in sorted(Path(root).rglob("scene.json")):
        if scene.parent.name.startswith("v1.0") and (scene.parent / "sample_data.json").exists():
            return scene.parent.parent, scene.parent.name
    return None


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXT and p.is_file())


def detect_kind(root: Path) -> str:
    """Nhận dạng loại dữ liệu trong thư mục đã giải nén / thư mục upload."""
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    if len(files) == 1 and files[0].suffix.lower() in VIDEO_EXT:
        return "video"
    if find_nuscenes(root):
        return "nuscenes"
    if lidarcam.detect(root):
        return "lidar_cam"
    if kitti.detect(root):
        return "kitti"
    if any(p.suffix.lower() in VIDEO_EXT for p in files):
        return "video"
    if image_files(root):
        return "images"
    raise IngestError(
        "Không nhận ra dữ liệu. Hỗ trợ: video mp4, ảnh (jpg/png hoặc zip ảnh), zip nuScenes, zip KITTI, "
        "zip LiDAR + camera (calib.json + lidar/ + thư mục camera)"
    )
