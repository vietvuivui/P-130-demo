"""Homography mặt đường và ghép ảnh BEV (src/services/bev.py)."""

import numpy as np

from src.models.schemas3d import Camera3D, Frame3DRecord
from src.services.bev import bev_camera, bev_mosaic, fit_ground, ground_height, ground_homography, lidar_uvd_to_ego

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


def _frame():
    cam = Camera3D(sd_token="c", path="x.jpg", intrinsic=K.tolist(), cam_from_lidar=CAM_FROM_LIDAR.tolist())
    return Frame3DRecord(frame_id="f", model="m", sample_token="t", scene="s", index=0, lidar_sd_token="l",
                         global_from_lidar=np.eye(4).tolist(), cameras={"CAM_FRONT": cam})  # fmt: skip


def _road(rng, n=20000, slope=0.0):
    x, y = rng.uniform(-30, 30, n), rng.uniform(-30, 30, n)
    return np.c_[x, y, -1.8 + slope * x + rng.normal(0, 0.02, n), np.zeros(n)]


def test_fit_ground_follows_slope():
    rng = np.random.default_rng(1)
    car = np.c_[rng.uniform(8, 12, 600), rng.uniform(-1, 1, 600), rng.uniform(-1.8, -0.3, 600), np.zeros(600)]
    gm = fit_ground(np.r_[_road(rng, slope=0.04), car])
    assert abs(gm.coef[0] - 0.04) < 0.01
    for x in (-20.0, 5.0, 25.0):
        assert abs(gm.height(np.array([x]), np.array([3.0]))[0] - (-1.8 + 0.04 * x)) < 0.1


def test_mosaic_blanks_ground_hidden_behind_wall():
    rng = np.random.default_rng(2)
    wall = np.c_[np.full(3000, 10.0), rng.uniform(-3, 3, 3000), rng.uniform(-1.8, 1.0, 3000), np.zeros(3000)]
    img = np.full((900, 1600, 3), 128, np.uint8)
    out = bev_mosaic({"CAM_FRONT": img}, _frame(), -1.8, range_m=20, res=0.5, points=np.r_[_road(rng), wall])
    row = lambda y: int((20 - y) / 0.5)  # noqa: E731
    col = lambda x: int((x + 20) / 0.5)  # noqa: E731
    assert out[row(0), col(6), 3] > 0  # trước tường: thấy mặt đường
    assert out[row(0), col(16), 3] == 0  # sau tường cao hơn camera: bị che
    assert out[row(9), col(16), 3] > 0  # lệch khỏi tường: vẫn thấy
    plain = bev_mosaic({"CAM_FRONT": img}, _frame(), -1.8, range_m=20, res=0.5)
    assert plain[row(0), col(16), 3] > 0  # không có LiDAR: IPM thuần như cũ


def test_bev_camera_uses_projected_lidar():
    # camera hành trình cao 1.5 m nhìn theo x của ego; cột cao 3 m cách 8 m phía trước
    from src.services.bev import nominal_cam_from_ego

    cfe = nominal_cam_from_ego(1.5)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(2, 50, 8000), rng.uniform(-20, 20, 8000)
    ground = np.c_[x, y, np.zeros_like(x)]
    pole = np.c_[np.full(800, 8.0), rng.uniform(-1, 1, 800), rng.uniform(0, 3, 800)]
    pts = np.r_[ground, pole]
    cam = pts @ cfe[:3, :3].T + cfe[:3, 3]
    cam = cam[cam[:, 2] > 1]
    uv = cam @ K.T
    lidar = {"u": (uv[:, 0] / uv[:, 2]).tolist(), "v": (uv[:, 1] / uv[:, 2]).tolist(), "d": cam[:, 2].tolist()}
    ego = lidar_uvd_to_ego(lidar, K, cfe)
    assert np.allclose(ego[:, 2].min(), 0, atol=1e-6)
    img = np.full((900, 1600, 3), 200, np.uint8)
    out = bev_camera(img, K, cfe, points_ego=ego)
    r = lambda xf: int((50 - xf) / 0.1)  # noqa: E731
    c = lambda yl: int((20 - yl) / 0.1)  # noqa: E731
    assert out[r(5), c(0), 3] > 0
    assert out[r(20), c(0), 3] == 0  # sau cột
    assert out[r(20), c(8), 3] > 0


def test_mosaic_fills_hidden_ground_from_neighbor_keyframe():
    # frame này: tường ở x = 10 che mặt đường phía sau. Keyframe sau: xe đã qua tường (x = 11), nhìn thấy x = 16.
    rng = np.random.default_rng(4)
    wall = np.c_[np.full(3000, 10.0), rng.uniform(-3, 3, 3000), rng.uniform(-1.8, 1.0, 3000), np.zeros(3000)]
    pts = np.r_[_road(rng), wall]
    nb = _frame()
    move = np.eye(4)
    move[0, 3] = 11.0
    nb.global_from_lidar = move.tolist()
    nb_pts = pts - np.array([11.0, 0, 0, 0])
    img = np.full((900, 1600, 3), 128, np.uint8)
    img_nb = np.full((900, 1600, 3), 60, np.uint8)
    row = lambda y: int((20 - y) / 0.5)  # noqa: E731
    col = lambda x: int((x + 20) / 0.5)  # noqa: E731
    out = bev_mosaic({"CAM_FRONT": img}, _frame(), -1.8, range_m=20, res=0.5, points=pts,
                     neighbors=[({"CAM_FRONT": img_nb}, nb, nb_pts)])  # fmt: skip
    assert out[row(0), col(16), 3] > 0 and out[row(0), col(16), 0] == 60  # lấy từ keyframe sau
    assert out[row(0), col(6), 0] == 128  # frame này thấy rõ hơn (gần hơn keyframe sau không thấy chỗ này)


def test_bev_camera_fills_from_later_keyframe():
    # cột ở x = 8 che mặt đường phía sau; keyframe sau (xe đã tiến 10 m) nhìn gần chỗ đó
    from src.services.bev import nominal_cam_from_ego

    cfe = nominal_cam_from_ego(1.5)
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-10, 60, 20000), rng.uniform(-20, 20, 20000)
    ground = np.c_[x, y, np.zeros_like(x)]
    wall = np.c_[np.full(1500, 8.0), rng.uniform(-3, 3, 1500), rng.uniform(0, 3, 1500)]
    pts = np.r_[ground, wall]
    move = np.eye(4)
    move[0, 3] = -10.0  # ego sau <- ego này: điểm lùi lại 10 m
    img, img_nb = np.full((900, 1600, 3), 200, np.uint8), np.full((900, 1600, 3), 90, np.uint8)
    out = bev_camera(img, K, cfe, points_ego=pts, neighbors=[(img_nb, K, cfe, move, pts + [-10.0, 0, 0])])
    r = lambda xf: int((50 - xf) / 0.1)  # noqa: E731
    c = lambda yl: int((20 - yl) / 0.1)  # noqa: E731
    assert out[r(20), c(0), 3] > 0 and abs(int(out[r(20), c(0), 0]) - 90) <= 2  # sau cột: lấy từ keyframe sau
    assert out[r(20), c(8), 3] > 0
