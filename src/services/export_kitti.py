"""Xuất nhãn 3D đã duyệt sang định dạng KITTI object (FR-17), đọc được bằng KITTI devkit / nuscenes-devkit (KittiDB),
MMDetection3D, OpenPCDet.

    training/image_2/000000.png     ảnh camera (mặc định CAM_FRONT), PNG như KITTI
    training/velodyne/000000.bin    point cloud keyframe, float32 x y z reflectance (0-1), hệ Velodyne KITTI (x trước, y trái)
    training/calib/000000.txt       P0-P3 = [K | 0], R0_rect = I, Tr_velo_to_cam, Tr_imu_to_velo = I
    training/label_2/000000.txt     type trunc occ alpha x1 y1 x2 y2 h w l x y z ry  (hệ camera, tâm đáy box)
    ImageSets/train.txt             danh sách chỉ số
    frames.csv                      chỉ số KITTI <-> frame_id / sample_token / scene của AutoLabel

Nhãn KITTI chỉ gồm vật nằm trong ảnh của camera đã chọn (quy ước KITTI); tên lớp giữ tên lớp nuScenes detection (car,
pedestrian, traffic_cone...) như `export_kitti.py` của nuscenes-devkit.
"""

from __future__ import annotations

import csv
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.schemas3d import Frame3DRecord, Object3D
from src.services.store import WorkspaceStore

# Velodyne KITTI (x trước, y trái, z lên) <- LIDAR_TOP nuScenes (x phải, y trước, z lên): quay -90° quanh z
R_VELO_FROM_NUSC = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def box_corners(center, size, yaw) -> np.ndarray:
    """8 đỉnh (8, 3) của box trong hệ LiDAR; size = [w, l, h], l dọc hướng yaw."""
    w, length, h = size
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * x - s * y, s * x + c * y, z], axis=1) + np.asarray(center)


def _bbox2d(corners_cam: np.ndarray, kmat: np.ndarray, w: int, h: int) -> tuple[list[float], float] | None:
    """Box 2D (cắt theo ảnh) của 8 đỉnh trong hệ camera, và độ bị cắt (truncation) 0-1. None nếu không lọt vào ảnh."""
    near = 0.1
    front = corners_cam[:, 2] > near
    if not front.any():
        return None
    pts = [corners_cam[i] for i in range(8) if front[i]]
    for i, j in EDGES:  # cạnh cắt mặt phẳng gần: lấy giao điểm để box 2D không bị vô hạn
        if front[i] != front[j]:
            t = (near - corners_cam[i, 2]) / (corners_cam[j, 2] - corners_cam[i, 2])
            pts.append(corners_cam[i] + t * (corners_cam[j] - corners_cam[i]))
    p = np.array(pts)
    uv = (kmat @ p.T)[:2] / p[:, 2]
    x1, y1 = uv.min(axis=1)
    x2, y2 = uv.max(axis=1)
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, w - 1.0), min(y2, h - 1.0)
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
        return None
    full = max((x2 - x1) * (y2 - y1), 1e-9)
    trunc = float(np.clip(1 - (cx2 - cx1) * (cy2 - cy1) / full, 0, 1))
    return [cx1, cy1, cx2, cy2], trunc


def _occlusion(o: Object3D) -> int:
    occ = o.verify.occlusion if o.verify else None
    if occ is None:
        return 3  # không rõ
    return 0 if occ < 0.25 else 1 if occ < 0.6 else 2


def label_line(o: Object3D, cam_from_lidar: np.ndarray, kmat: np.ndarray, w: int, h: int) -> str | None:
    """Một dòng label_2 KITTI cho box, hoặc None nếu box không nằm trong ảnh."""
    center, size, yaw = np.asarray(o.box.center, float), o.box.size, o.box.yaw
    rot, t = cam_from_lidar[:3, :3], cam_from_lidar[:3, 3]
    got = _bbox2d(box_corners(center, size, yaw) @ rot.T + t, kmat, w, h)
    if got is None:
        return None
    (x1, y1, x2, y2), trunc = got
    bottom = rot @ (center - np.array([0.0, 0.0, size[2] / 2])) + t  # KITTI: toạ độ tâm đáy box trong hệ camera
    d = rot @ np.array([math.cos(yaw), math.sin(yaw), 0.0])  # hướng đầu xe trong hệ camera
    ry = _wrap(-math.atan2(d[2], d[0]))
    alpha = _wrap(ry - math.atan2(bottom[0], bottom[2]))
    name = o.review.final_label or o.label
    wd, length, hh = size
    return (
        f"{name} {trunc:.2f} {_occlusion(o)} {alpha:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{hh:.2f} {wd:.2f} {length:.2f} {bottom[0]:.2f} {bottom[1]:.2f} {bottom[2]:.2f} {ry:.2f}"
    )


def _orthonormal(mat: np.ndarray) -> np.ndarray:
    """Phần quay của ma trận ngoại tham số được lưu làm tròn 6 chữ số: chuẩn hoá lại thành ma trận quay đúng (SVD), vì
    các bộ đọc KITTI (vd. KittiDB đổi sang quaternion) đòi ma trận trực giao chặt."""
    u, _, vt = np.linalg.svd(mat[:3, :3])
    out = mat.copy()
    out[:3, :3] = u @ vt
    return out


def _calib_text(kmat: np.ndarray, cam_from_velo: np.ndarray) -> str:
    proj = np.hstack([kmat, np.zeros((3, 1))])
    fmt = lambda m: " ".join(f"{v:.12e}" for v in np.asarray(m, float).ravel())  # noqa: E731
    lines = [f"proj{i}: {fmt(proj)}" for i in range(4)]
    lines += [f"R0_rect: {fmt(np.eye(3))}", f"Tr_velo_to_cam: {fmt(cam_from_velo[:3])}",
              f"Tr_imu_to_velo: {fmt(np.eye(4)[:3])}"]  # fmt: skip
    return "\n".join(lines) + "\n"


def _lidar_points(store: WorkspaceStore, data, f: Frame3DRecord) -> np.ndarray:
    """Point cloud keyframe (N, 4) hệ LiDAR nuScenes, cường độ 0-255: file gốc nếu có, không thì bản rút gọn của UI."""
    if data is not None:
        sd = data.sample_data.get(f.lidar_sd_token)
        if sd:
            path = Path(data.dataroot) / sd["filename"]
            if path.exists():
                raw = np.fromfile(path, np.float32)
                cols = 5 if raw.size % 5 == 0 else 4
                return raw.reshape(-1, cols)[:, :4]
    path = store.points_path(f.frame_id)
    return np.fromfile(path, np.float32).reshape(-1, 4) if path.exists() else np.zeros((0, 4), np.float32)


def export_kitti(store: WorkspaceStore, dataroot: Path, version: str, model: str, out_zip: Path,
                 camera: str = "CAM_FRONT", include_pending: bool = False, image_loader=None) -> dict:  # fmt: skip
    """image_loader(path) -> PIL.Image: cho phép làm mờ mặt / biển số trước khi ghi ảnh (FR-03)."""
    frames = [
        f for f in store.list_frames3d(model) if f.status == "approved" or (include_pending and f.status != "rejected")
    ]
    frames = [f for f in frames if camera in f.cameras]
    if not frames:
        raise ValueError(f"Chưa có keyframe 3D nào được approve (có camera {camera})")
    try:
        from src.services.nuscenes_data import NuScenesMini

        data = NuScenesMini(dataroot, version)
    except Exception:  # thiếu bảng: dùng point cloud rút gọn trong workspace
        data = None

    stage = out_zip.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    tr = stage / "training"
    for d in ("image_2", "velodyne", "calib", "label_2"):
        (tr / d).mkdir(parents=True)
    velo4 = np.eye(4)
    velo4[:3, :3] = R_VELO_FROM_NUSC
    rows, n_labels, skipped = [], 0, 0
    for i, f in enumerate(sorted(frames, key=lambda f: (f.scene, f.index))):
        idx = f"{i:06d}"
        cam = f.cameras[camera]
        kmat = np.asarray(cam.intrinsic, float)
        c_from_l = _orthonormal(np.asarray(cam.cam_from_lidar, float))
        (tr / "calib" / f"{idx}.txt").write_text(_calib_text(kmat, c_from_l @ np.linalg.inv(velo4)))
        src = Path(dataroot) / cam.path
        img = image_loader(src) if image_loader else Image.open(src)
        img.convert("RGB").save(tr / "image_2" / f"{idx}.png")
        pts = _lidar_points(store, data, f)
        out = np.zeros((len(pts), 4), np.float32)
        out[:, :3] = pts[:, :3] @ R_VELO_FROM_NUSC.T
        out[:, 3] = np.clip(pts[:, 3] / 255.0, 0, 1) if len(pts) and pts[:, 3].max() > 1.0 else pts[:, 3]
        out.tofile(tr / "velodyne" / f"{idx}.bin")
        lines = []
        for o in f.objects:
            keep = o.review.status == "approved" or (include_pending and o.review.status == "pending")
            if not keep:
                continue
            line = label_line(o, c_from_l, kmat, cam.width, cam.height)
            if line is None:
                skipped += 1  # ngoài ảnh camera này
                continue
            lines.append(line)
        (tr / "label_2" / f"{idx}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_labels += len(lines)
        rows.append([idx, f.frame_id, f.sample_token, f.scene, f.index, f.status, len(lines)])
    (stage / "ImageSets").mkdir()
    (stage / "ImageSets" / "train.txt").write_text("\n".join(r[0] for r in rows) + "\n")
    with open(stage / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kitti_index", "frame_id", "sample_token", "scene", "index", "status", "labels"])
        w.writerows(rows)
    (stage / "README.txt").write_text(
        f"Nhãn 3D xuất từ AutoLabel 3D, định dạng KITTI object (camera {camera}).\n"
        "Tên lớp là tên lớp nuScenes detection (như export_kitti.py của nuscenes-devkit). Nhãn chỉ gồm vật nằm trong ảnh.\n"
        "velodyne/*.bin: hệ Velodyne KITTI (x trước, y trái, z lên), reflectance 0-1. frames.csv: chỉ số <-> frame_id.\n",
        encoding="utf-8",
    )
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in stage.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    shutil.rmtree(stage)
    return dict(file=out_zip.name, frames=len(rows), labels=n_labels, outside_camera=skipped, camera=camera)
