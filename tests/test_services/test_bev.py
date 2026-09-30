"""Homography mặt đường và ghép ảnh BEV (src/services/bev.py)."""

import numpy as np

from src.models.schemas3d import Camera3D, Frame3DRecord
from src.services.bev import bev_mosaic, ground_height, ground_homography

K = np.array([[1000.0, 0, 800], [0, 1000.0, 450], [0, 0, 1]])
# camera nhìn theo trục x của LiDAR, đặt ngay tại gốc LiDAR
CAM_FROM_LIDAR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)


def test_homography_matches_full_projection():
    z0 = -1.8
    homo = ground_homography(K, CAM_FROM_LIDAR, z0)
    for x, y in [(5, 0), (12, -3), (30, 4)]:
        p = CAM_FROM_LIDAR @ np.array([x, y, z0, 1.0])
        uv = K @ p[:3]
        h = homo @ np.array([x, y, 1.0])
        assert np.allclose(uv[:2] / uv[2], h[:2] / h[2])


def test_ground_height_from_points():
    rng = np.random.default_rng(0)
    ang, r = rng.uniform(0, 2 * np.pi, 5000), rng.uniform(4, 24, 5000)
    road = np.c_[r * np.cos(ang), r * np.sin(ang), rng.normal(-1.9, 0.02, 5000), np.zeros(5000)]
    wall = np.c_[rng.uniform(5, 20, 800), np.full(800, 6.0), rng.uniform(-1.9, 2, 800), np.zeros(800)]
    assert abs(ground_height(np.r_[road, wall]) + 1.9) < 0.05
    assert ground_height(np.zeros((0, 4))) == -1.84


def test_mosaic_puts_front_camera_in_front():
    cam = Camera3D(sd_token="c", path="x.jpg", intrinsic=K.tolist(), cam_from_lidar=CAM_FROM_LIDAR.tolist())
    frame = Frame3DRecord(frame_id="f", model="m", sample_token="t", scene="s", index=0, lidar_sd_token="l",
                          global_from_lidar=np.eye(4).tolist(), cameras={"CAM_FRONT": cam})  # fmt: skip
    img = np.zeros((900, 1600, 3), np.uint8)
    img[:, :800] = [0, 255, 0]  # nửa trái ảnh (phía y > 0 của LiDAR)
    img[:, 800:] = [0, 0, 255]
    bev = bev_mosaic({"CAM_FRONT": img}, frame, -1.8, range_m=20, res=0.5)
    # hàng 0 = y = +20, cột 0 = x = -20; ô (x=10, y=+3) nằm bên trái trục -> xanh lá, (x=10, y=-3) -> xanh dương
    row = lambda y: int((20 - y) / 0.5)  # noqa: E731
    col = lambda x: int((x + 20) / 0.5)  # noqa: E731
    assert tuple(bev[row(3), col(10)]) == (0, 255, 0, 255)
    assert tuple(bev[row(-3), col(10)]) == (0, 0, 255, 255)
    assert bev[row(0), col(-10), 3] == 0  # sau lưng camera
