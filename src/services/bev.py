"""Ảnh nhìn từ trên xuống (BEV) ghép từ 6 camera bằng homography mặt đường (inverse perspective mapping).

Giả sử mặt đường là mặt phẳng z = z0 trong hệ LiDAR của keyframe. Với camera có ma trận nội K và ngoại
[R | t] (cam_from_lidar), một điểm mặt đường (x, y, z0) chiếu xuống ảnh theo

    s · [u, v, 1]^T = K · [r1  r2  z0·r3 + t] · [x, y, 1]^T = H · [x, y, 1]^T

nên H (3x3) là homography giữa mặt đường và ảnh. Mỗi ô của lưới BEV lấy màu từ camera nhìn ô đó gần trục quang học
nhất. Chỉ đúng cho những gì nằm trên mặt đường (vạch kẻ, lề, chân xe); vật cao bị kéo dài ra xa camera, đó là giới
hạn vốn có của IPM, không phải lỗi.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from src.models.schemas3d import Frame3DRecord

GROUND_DEFAULT = -1.84  # LiDAR nuScenes đặt cao ~1.84 m so với mặt đường


def ground_height(points: np.ndarray, r_min: float = 3.0, r_max: float = 25.0) -> float:
    """Độ cao mặt đường trong hệ LiDAR: mode của z các điểm thấp quanh xe (mặt đường chiếm đa số điểm thấp)."""
    if points is None or len(points) == 0:
        return GROUND_DEFAULT
    r = np.hypot(points[:, 0], points[:, 1])
    z = points[(r > r_min) & (r < r_max), 2]
    z = z[(z > -3.5) & (z < -0.5)]
    if len(z) < 50:
        return GROUND_DEFAULT
    hist, edges = np.histogram(z, bins=np.arange(-3.5, -0.5 + 1e-9, 0.05))
    k = int(np.argmax(hist))
    near = z[(z >= edges[max(k - 1, 0)]) & (z < edges[min(k + 2, len(edges) - 1)])]
    return round(float(np.median(near)), 3)


def ground_homography(intrinsic: np.ndarray, cam_from_lidar: np.ndarray, z0: float) -> np.ndarray:
    """H: [x, y, 1] trên mặt đường z = z0 (hệ LiDAR) -> [u·s, v·s, s] trên ảnh (s = độ sâu theo trục camera)."""
    rot, t = cam_from_lidar[:3, :3], cam_from_lidar[:3, 3]
    return intrinsic @ np.column_stack([rot[:, 0], rot[:, 1], z0 * rot[:, 2] + t])


def bev_mosaic(images: dict[str, np.ndarray], frame: Frame3DRecord, z0: float, range_m: float = 40.0,
               res: float = 0.1, min_depth: float = 1.0, fade_from: float = 25.0) -> np.ndarray:  # fmt: skip
    """Ảnh BEV RGBA (alpha 0 = không camera nào thấy). Hàng 0 là y = +range, cột 0 là x = -range: ảnh đặt thẳng lên
    mặt phẳng x-y của LiDAR (nuScenes: x sang phải, y về phía trước). Xa hơn `fade_from` m ảnh mờ dần, vì mỗi pixel
    camera phủ một vùng mặt đường lớn dần theo bình phương khoảng cách (vùng đó bị kéo nhoè)."""
    n = int(round(2 * range_m / res))
    c = (np.arange(n) + 0.5) * res - range_m
    xs, ys = np.meshgrid(c, c[::-1])  # cột: x tăng dần; hàng: y giảm dần
    ground = np.stack([xs.ravel(), ys.ravel(), np.ones(n * n)])
    out = np.zeros((n * n, 4), np.uint8)
    best = np.full(n * n, np.inf)  # |góc lệch trục quang học|, nhỏ hơn = ít méo hơn
    inside_range = np.hypot(xs.ravel(), ys.ravel()) <= range_m
    for name, img in images.items():
        cam = frame.cameras.get(name)
        if cam is None or img is None:
            continue
        intr = np.asarray(cam.intrinsic, float)
        homo = ground_homography(intr, np.asarray(cam.cam_from_lidar, float), z0)
        uvs = homo @ ground
        depth = uvs[2]
        ok = (depth > min_depth) & inside_range
        u = np.full(n * n, -1.0)
        v = np.full(n * n, -1.0)
        u[ok], v[ok] = uvs[0, ok] / depth[ok], uvs[1, ok] / depth[ok]
        h, w = img.shape[:2]
        ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        off_axis = np.abs(u - intr[0, 2]) / intr[0, 0]
        take = ok & (off_axis < best)
        best[take] = off_axis[take]
        out[take, :3] = img[v[take].astype(int), u[take].astype(int), :3]
        out[take, 3] = 255
    r = np.hypot(xs.ravel(), ys.ravel())
    far = (out[:, 3] > 0) & (r > fade_from)
    out[far, 3] = np.clip(255 - (r[far] - fade_from) / max(range_m - fade_from, 1e-6) * 175, 80, 255).astype(np.uint8)
    return out.reshape(n, n, 4)


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except OSError:
        return None


def to_png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------------------------------------------------------------- BEV cho chế độ 2D (một camera, hệ ego)
# Hệ ego nuScenes: gốc ở mặt đường dưới trục sau, x về phía trước, y sang trái, z lên -> mặt đường là z = 0.
NOMINAL_CAM_HEIGHT = 1.5  # m, video tải lên không có calibration: giả định camera hành trình cao 1.5 m, nhìn thẳng
BEV2D_X = (0.0, 50.0)  # m phía trước
BEV2D_Y = (-20.0, 20.0)  # m ngang (dương = bên trái)


def nominal_cam_from_ego(height: float = NOMINAL_CAM_HEIGHT) -> np.ndarray:
    """Camera đặt tại (0, 0, height) của hệ ego, trục quang học song song mặt đường theo hướng x."""
    rot = np.array(
        [[0.0, -1, 0], [0, 0, -1], [1, 0, 0]]
    )  # ego (x trước, y trái, z lên) -> camera (x phải, y xuống, z trước)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = -rot @ np.array([0.0, 0.0, height])
    return m


def bev_camera(img: np.ndarray, intrinsic: np.ndarray, cam_from_ego: np.ndarray, x_range=BEV2D_X, y_range=BEV2D_Y,
               res: float = 0.1, min_depth: float = 1.0, fade_from: float = 25.0) -> np.ndarray:  # fmt: skip
    """Ảnh một camera chiếu xuống mặt đường ego (z = 0), RGBA. Hàng 0 = x xa nhất (phía trước ở trên), cột 0 = y lớn
    nhất (bên trái ở bên trái): nhìn từ trên xuống, đầu xe hướng lên."""
    xs = x_range[1] - (np.arange(int(round((x_range[1] - x_range[0]) / res))) + 0.5) * res
    ys = y_range[1] - (np.arange(int(round((y_range[1] - y_range[0]) / res))) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    homo = ground_homography(intrinsic, cam_from_ego, 0.0)
    uvs = homo @ np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    depth = uvs[2]
    ok = depth > min_depth
    u = np.where(ok, uvs[0] / np.where(ok, depth, 1), -1)
    v = np.where(ok, uvs[1] / np.where(ok, depth, 1), -1)
    h, w = img.shape[:2]
    ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    out = np.zeros((gx.size, 4), np.uint8)
    out[ok, :3] = img[v[ok].astype(int), u[ok].astype(int), :3]
    dist = np.hypot(gx.ravel(), gy.ravel())
    out[ok, 3] = np.clip(255 - np.maximum(dist[ok] - fade_from, 0) / max(x_range[1] - fade_from, 1e-6) * 175, 80, 255)
    return out.reshape(gx.shape[0], gx.shape[1], 4)
