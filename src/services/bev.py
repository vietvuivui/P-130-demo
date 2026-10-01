"""Ảnh nhìn từ trên xuống (BEV) ghép từ 6 camera (chế độ 3D) hoặc từ một camera (chế độ Ảnh / Video).

Mỗi ô của lưới BEV là một điểm mặt đường; chiếu điểm đó vào camera để lấy màu (inverse perspective mapping). Với mặt
phẳng z = z0 phép chiếu là một homography

    s · [u, v, 1]^T = K · [r1  r2  z0·r3 + t] · [x, y, 1]^T = H · [x, y, 1]^T

IPM thuần chỉ đúng cho mặt đường: mọi vật cao (xe, cây, tường) bị kéo thành vệt dài toả ra từ camera, và khi đường dốc
thì ảnh ở xa trượt khỏi point cloud. Khi có LiDAR, hai lỗi đó được sửa:

1. Mặt đường lấy từ LiDAR (`fit_ground`: mặt phẳng RANSAC + lưới sai lệch mọc vùng) thay vì một độ cao cố định.
2. Ô mặt đường mà camera không thực sự nhìn thấy — tia từ camera tới nó bị một vật cao hơn mặt đường chắn
   (`shadow_map`) — không lấy màu từ camera đó. Chỗ đó để trống, point cloud bên dưới hiện ra.

Vùng hai camera cùng thấy được trộn mềm theo khoảng cách tới tâm ảnh; lấy mẫu song tuyến.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.models.schemas3d import Frame3DRecord

GROUND_DEFAULT = -1.84  # LiDAR nuScenes đặt cao ~1.84 m so với mặt đường
BEV_VERSION = 3  # đổi cách dựng ảnh thì tăng: ảnh cache cũ không được dùng lại
BEV_NEIGHBORS = 4  # ghép thêm mặt đường từ ±4 keyframe (±2 s) cùng scene


def ground_height(points: np.ndarray, r_min: float = 3.0, r_max: float = 25.0,
                  z_range: tuple[float, float] = (-3.5, -0.5)) -> float:  # fmt: skip
    """Độ cao mặt đường: mode của z các điểm thấp quanh gốc toạ độ (mặt đường chiếm đa số điểm thấp). `z_range`: khoảng
    tìm, mặc định cho hệ LiDAR nuScenes (LiDAR cao ~1.84 m); hệ ego (mặt đường z ≈ 0) dùng (-1.5, 1.5)."""
    default = GROUND_DEFAULT if z_range[0] < GROUND_DEFAULT < z_range[1] else float(np.mean(z_range))
    if points is None or len(points) == 0:
        return default
    r = np.hypot(points[:, 0], points[:, 1])
    z = points[(r > r_min) & (r < r_max), 2]
    z = z[(z > z_range[0]) & (z < z_range[1])]
    if len(z) < 50:
        return default
    hist, edges = np.histogram(z, bins=np.arange(z_range[0], z_range[1] + 1e-9, 0.05))
    k = int(np.argmax(hist))
    near = z[(z >= edges[max(k - 1, 0)]) & (z < edges[min(k + 2, len(edges) - 1)])]
    return round(float(np.median(near)), 3)


def ground_homography(intrinsic: np.ndarray, cam_from_lidar: np.ndarray, z0: float) -> np.ndarray:
    """H: [x, y, 1] trên mặt đường z = z0 (hệ LiDAR) -> [u·s, v·s, s] trên ảnh (s = độ sâu theo trục camera)."""
    rot, t = cam_from_lidar[:3, :3], cam_from_lidar[:3, 3]
    return intrinsic @ np.column_stack([rot[:, 0], rot[:, 1], z0 * rot[:, 2] + t])


class GroundModel:
    """Mặt đường trong hệ LiDAR: mặt phẳng z = a·x + b·y + c (RANSAC, bắt được xe đang lên / xuống dốc, LiDAR nghiêng)
    cộng một lưới sai lệch mượt ô `cell` m (đường cong, gờ, lề). Một độ cao cố định làm mặt đường ở xa lệch hàng chục
    cm, ảnh chiếu xuống bị trượt so với point cloud."""

    def __init__(self, coef: np.ndarray, resid: np.ndarray | None = None, extent: float = 50.0, cell: float = 2.0):
        self.coef, self.resid, self.extent, self.cell = np.asarray(coef, float), resid, extent, cell

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = self.coef[0] * x + self.coef[1] * y + self.coef[2]
        if self.resid is None:
            return z
        n = self.resid.shape[0]
        gi = np.clip((x + self.extent) / self.cell - 0.5, 0, n - 1.001)
        gj = np.clip((y + self.extent) / self.cell - 0.5, 0, n - 1.001)
        i0, j0 = gi.astype(int), gj.astype(int)
        fi, fj = gi - i0, gj - j0
        r = self.resid
        return z + (r[i0, j0] * (1 - fi) * (1 - fj) + r[i0 + 1, j0] * fi * (1 - fj)
                    + r[i0, j0 + 1] * (1 - fi) * fj + r[i0 + 1, j0 + 1] * fi * fj)  # fmt: skip


def fit_ground(points: np.ndarray | None, extent: float = 50.0, cell: float = 2.0, seed: int = 0,
               z_range: tuple[float, float] = (-3.5, -0.5), min_points: int = 500) -> GroundModel:  # fmt: skip
    """RANSAC mặt phẳng trên điểm thấp quanh xe, rồi lưới sai lệch: mỗi ô lấy phân vị 10% độ cao điểm (ô chỉ có nóc xe /
    tường bị loại vì lệch mặt phẳng > 0.4 m), lấp ô trống và làm mượt bằng tích chập chuẩn hoá. Thiếu điểm: mặt phẳng
    ngang ở độ cao `ground_height`."""
    from scipy.ndimage import gaussian_filter

    z_flat = ground_height(points, z_range=z_range)
    flat = GroundModel([0.0, 0.0, z_flat])
    if points is None or len(points) < min_points:
        return flat
    r = np.hypot(points[:, 0], points[:, 1])
    cand = points[(r > 2.5) & (r < extent) & (points[:, 2] > z_flat - 1.5) & (points[:, 2] < z_flat + 1.0), :3]
    if len(cand) < min_points * 0.6:
        return flat
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(150):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(nrm)
        if norm < 1e-6 or abs(nrm[2]) / norm < 0.985:  # nghiêng quá ~10°: không phải mặt đường
            continue
        nrm /= norm
        inl = np.abs((cand - s[0]) @ nrm) < 0.12
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < min_points * 0.4:
        return flat
    g = cand[best]
    coef = np.linalg.lstsq(np.c_[g[:, 0], g[:, 1], np.ones(len(g))], g[:, 2], rcond=None)[0]
    plane = GroundModel(coef, extent=extent, cell=cell)
    n = int(round(2 * extent / cell))
    near = points[(np.abs(points[:, 0]) < extent) & (np.abs(points[:, 1]) < extent)]
    near = near[np.hypot(near[:, 0], near[:, 1]) > 2.5]
    ii = ((near[:, 0] + extent) / cell).astype(int).clip(0, n - 1)
    jj = ((near[:, 1] + extent) / cell).astype(int).clip(0, n - 1)
    dz = near[:, 2] - plane.height(near[:, 0], near[:, 1])
    order = np.lexsort((dz, jj, ii))
    key = ii[order] * n + jj[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(key)])
    q = dz[order][starts + (counts * 0.1).astype(int)]  # phân vị 10% mỗi ô
    ci, cj = key[starts] // n, key[starts] % n
    surf = np.zeros((n, n))
    # Mọc vùng: lần đầu nhận ô gần mặt phẳng, các lần sau nhận ô gần mặt đã làm mượt, nên dốc thoai thoải ở xa (đường
    # lên đồi) vẫn được coi là mặt đường; ô chỉ có nóc xe / tường lệch hẳn khỏi mặt xung quanh nên bị loại.
    for _ in range(4):
        good = (counts >= 3) & (np.abs(q - surf[ci, cj]) < 0.35)
        resid, wt = np.zeros((n, n)), np.zeros((n, n))
        resid[ci[good], cj[good]] = q[good]
        wt[ci[good], cj[good]] = 1.0
        sm_w = gaussian_filter(wt, 1.5)
        new = gaussian_filter(resid * wt, 1.5) / np.maximum(sm_w, 1e-6)
        # ô không có số liệu: giữ mặt lần trước (lần đầu là mặt phẳng)
        surf = np.where(sm_w < 1e-3, surf, new)
    return GroundModel(coef, np.clip(surf, -2.5, 2.5), extent, cell)


AZ_BIN = np.deg2rad(0.4)
RHO_BIN = 0.2


def shadow_map(points: np.ndarray, cam_xy: np.ndarray, cam_h: float, ground: GroundModel, max_rho: float,
               min_height: float = 0.5) -> np.ndarray | None:  # fmt: skip
    """Vùng mặt đường bị vật cao che khi nhìn từ camera, trên lưới cực (góc phương vị x khoảng cách ngang).

    Camera cao `cam_h` m. Một điểm LiDAR cao h (so với mặt đường) ở khoảng cách ngang ρ_o chắn tia tới mặt đường ở
    khoảng cách ρ khi h ≥ cam_h·(1 − ρ_o/ρ), tức với ρ_o < ρ ≤ ρ_o / (1 − h/cam_h) — vật thấp đổ bóng ngắn, vật cao
    bằng camera trở lên (xe tải, tường) đổ bóng tới hết tầm. Gộp bóng của mọi điểm bằng mảng hiệu theo ρ, rồi nới
    ±2 ô phương vị để lấp khe giữa các tia LiDAR."""
    from scipy.ndimage import maximum_filter1d

    hgt = points[:, 2] - ground.height(points[:, 0], points[:, 1])
    sel = (hgt > min_height) & (hgt < cam_h + 3.0)
    p, h = points[sel], hgt[sel]
    if not len(p):
        return None
    # LiDAR thường không chạm phần trên của vật ở gần (nóc xe cạnh bên): mỗi điểm lấy độ cao lớn nhất của vật quanh nó
    # (ô 0.4 m, nới 1 ô), cộng 0.1 m, để bóng phủ cả phần thân xe camera thấy mà LiDAR bỏ sót.
    cell = 0.4
    ci = np.floor(p[:, 0] / cell).astype(np.int64)
    cj = np.floor(p[:, 1] / cell).astype(np.int64)
    ci0, cj0 = ci.min(), cj.min()
    grid = np.zeros((ci.max() - ci0 + 1, cj.max() - cj0 + 1))
    np.maximum.at(grid, (ci - ci0, cj - cj0), h)
    from scipy.ndimage import maximum_filter

    h = maximum_filter(grid, size=3)[ci - ci0, cj - cj0] + 0.1
    dx, dy = p[:, 0] - cam_xy[0], p[:, 1] - cam_xy[1]
    rho = np.hypot(dx, dy)
    keep = (rho > 0.3) & (rho < max_rho)
    rho, h, dx, dy = rho[keep], h[keep], dx[keep], dy[keep]
    n_az, n_rho = int(np.ceil(2 * np.pi / AZ_BIN)), int(np.ceil(max_rho / RHO_BIN)) + 2
    az = ((np.arctan2(dy, dx) + np.pi) / AZ_BIN).astype(int) % n_az
    ratio = np.clip(h / cam_h, 0, 0.999)
    end = np.where(h >= cam_h, np.inf, rho / (1 - ratio))
    i0 = np.minimum((rho / RHO_BIN).astype(int) + 1, n_rho - 1)
    i1 = np.minimum(np.where(np.isinf(end), n_rho - 1, end / RHO_BIN).astype(int) + 1, n_rho - 1)
    diff = np.zeros((n_az, n_rho), np.int32)
    np.add.at(diff, (az, i0), 1)
    np.add.at(diff, (az, i1), -1)
    shadow = np.cumsum(diff, axis=1) > 0
    return maximum_filter1d(shadow.astype(np.uint8), size=5, axis=0, mode="wrap").astype(bool)


def in_shadow(shadow: np.ndarray, cam_xy: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    dx, dy = gx - cam_xy[0], gy - cam_xy[1]
    az = ((np.arctan2(dy, dx) + np.pi) / AZ_BIN).astype(int) % shadow.shape[0]
    ri = np.minimum((np.hypot(dx, dy) / RHO_BIN).astype(int), shadow.shape[1] - 1)
    return shadow[az, ri]


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    a, b = img[v0, u0].astype(np.float32), img[v0, u0 + 1].astype(np.float32)
    c, d = img[v0 + 1, u0].astype(np.float32), img[v0 + 1, u0 + 1].astype(np.float32)
    return (a * (1 - du) + b * du) * (1 - dv) + (c * (1 - du) + d * du) * dv


@dataclass
class _View:
    """Một lần chụp đem ghép: ảnh các camera (ảnh, intrinsic, camera <- hệ của frame đó), LiDAR và mặt đường trong hệ
    của frame đó, phép đổi hệ frame gốc -> frame đó, thứ tự (0 = frame gốc, 1 = frame lân cận gần nhất...)."""

    cams: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    points: np.ndarray | None
    ground: GroundModel | None
    from_ref: np.ndarray = field(default_factory=lambda: np.eye(4))
    order: int = 0


def _fuse(views: list[_View], ground_ref: np.ndarray, shadow_range: float, min_depth: float,
          body: tuple[float, float, float, float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Màu của các điểm mặt đường `ground_ref` (3, N, hệ frame gốc) từ mọi view. Trả về (rgb, seen, điểm LiDAR chạm mặt
    đường đổi về hệ gốc). `body` = (xmin, xmax, ymin, ymax) thân xe trong hệ frame: ô trong đó không lấy màu (camera thấy
    cốp / nắp capô của chính xe, chiếu xuống thành vệt; ghép nhiều frame thì vệt lặp thành sọc)."""
    n = ground_ref.shape[1]
    acc = np.zeros((n, 3), np.float32)
    wsum = np.zeros(n, np.float32)
    support = []
    multi = len(views) > 1
    for vw in views:
        g = ground_ref if vw.order == 0 else vw.from_ref[:3, :3] @ ground_ref + vw.from_ref[:3, 3:4]
        pts, gm = vw.points, vw.ground
        has_pts = pts is not None and len(pts) and gm is not None
        if has_pts:
            gp = pts[np.abs(pts[:, 2] - gm.height(pts[:, 0], pts[:, 1])) < 0.25, :3]
            back = np.linalg.inv(vw.from_ref)
            support.append((gp @ back[:3, :3].T + back[:3, 3])[:, :2])
        for img, intr, cfv in vw.cams:
            if img is None:
                continue
            uvs = intr @ (cfv[:3, :3] @ g + cfv[:3, 3:4])
            depth = uvs[2]
            h, w = img.shape[:2]
            ok = depth > min_depth
            u = np.where(ok, uvs[0] / np.where(ok, depth, 1), -1)
            v = np.where(ok, uvs[1] / np.where(ok, depth, 1), -1)
            ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            if body is not None and multi:
                ok &= ~((g[0] > body[0]) & (g[0] < body[1]) & (g[1] > body[2]) & (g[1] < body[3]))
            cam_pos = np.linalg.inv(cfv)[:3, 3]  # vị trí camera trong hệ của frame đó
            if has_pts:
                cam_h = max(float(cam_pos[2] - gm.height(cam_pos[:1], cam_pos[1:2])[0]), 0.5)
                shadow = shadow_map(pts, cam_pos[:2], cam_h, gm, shadow_range)
                if shadow is not None:
                    idx = np.flatnonzero(ok)
                    ok[idx[in_shadow(shadow, cam_pos[:2], g[0, idx], g[1, idx])]] = False
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            wt = np.clip(1.0 - np.abs(u[idx] - intr[0, 2]) / (w / 2), 0, 1) ** 2 + 1e-3
            if multi:
                # độ nét: mặt đường cách camera rho m -> mỗi pixel phủ ~rho^2 mặt đường; lũy thừa cao để lần nhìn gần
                # thắng hẳn (trộn trung bình các lần nhìn lệch nhau vài cm cũng làm nhoè)
                rho = np.hypot(g[0, idx] - cam_pos[0], g[1, idx] - cam_pos[1])
                wt = wt * (1.0 / (1.0 + (rho / 8.0) ** 2)) ** 6 * (0.85**vw.order)
            acc[idx] += _bilinear(img[..., :3], u[idx], v[idx]) * wt[:, None]
            wsum[idx] += wt
    seen = wsum > 0
    rgb = np.zeros((n, 3), np.float32)
    rgb[seen] = acc[seen] / wsum[seen, None]
    return rgb, seen, (np.concatenate(support) if support else np.zeros((0, 2)))


def _clean(seen2d: np.ndarray, rgb2d: np.ndarray, valid2d: np.ndarray, res: float, multi: bool,
           fill_holes: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Dọn vùng thấy được (lưới 2D): bỏ dải mảnh lọt giữa hai bóng và (khi ghép nhiều frame) mảnh rời < 4 m²; tô lỗ nhỏ
    (bóng của cột, người, rìa răng cưa; < `fill_holes` m²) từ xung quanh. Vùng lớn không ai thấy để trống."""
    from scipy.ndimage import binary_closing, binary_opening, label

    open_px = max(5, int(round(0.9 / res)) | 1) if multi else 5
    grid = binary_closing(binary_opening(seen2d, np.ones((open_px, open_px))), np.ones((3, 3))) & valid2d
    if multi:
        isl, ni = label(grid)
        if ni:
            grid &= (np.bincount(isl.ravel()) * res * res >= 4.0)[isl] & (isl > 0)
    if fill_holes > 0:
        holes, nh = label(~grid)
        if nh:
            small = np.bincount(holes.ravel()) * res * res < fill_holes
            small[0] = False
            for corner in (holes[0, 0], holes[0, -1], holes[-1, 0], holes[-1, -1]):
                small[corner] = False  # vùng bao ngoài
            fill = small[holes] & valid2d
            todo = fill | (grid & ~seen2d)
            if todo.any():
                import cv2

                img = np.clip(rgb2d, 0, 255).astype(np.uint8)
                img[~seen2d] = 0
                rgb2d = rgb2d.copy()
                rgb2d[todo] = cv2.inpaint(img, todo.astype(np.uint8), 3, cv2.INPAINT_TELEA)[todo]
                grid |= fill
    return grid, rgb2d


def _support_mask(sp: np.ndarray, x: np.ndarray, y: np.ndarray, dist0: np.ndarray, support_m: float,
                  cs: float = 0.5) -> np.ndarray:  # fmt: skip
    """Ô gần chỗ LiDAR chạm mặt đường (khoảng cách <= support_m + 5% khoảng cách tới xe). Sau tường / toà nhà / hàng rào
    kín LiDAR cũng không chạm đất, còn camera thì kéo vệt vật cao ra đó."""
    from scipy.ndimage import distance_transform_edt

    x0, y0 = min(x.min(), sp[:, 0].min()) - cs, min(y.min(), sp[:, 1].min()) - cs
    i = ((sp[:, 0] - x0) / cs).astype(int)
    j = ((sp[:, 1] - y0) / cs).astype(int)
    ci, cj = ((x - x0) / cs).astype(int), ((y - y0) / cs).astype(int)
    occ = np.zeros((max(i.max(), ci.max()) + 1, max(j.max(), cj.max()) + 1), bool)
    occ[i, j] = True
    return distance_transform_edt(~occ)[ci, cj] * cs <= support_m + 0.05 * dist0


def bev_mosaic(images: dict[str, np.ndarray], frame: Frame3DRecord, z0: float, range_m: float = 40.0,
               res: float = 0.1, min_depth: float = 1.0, fade_from: float = 25.0,
               points: np.ndarray | None = None, ground_model: GroundModel | None = None,
               neighbors: list[tuple[dict[str, np.ndarray], Frame3DRecord, np.ndarray | None]] | None = None,
               fill_holes: float = 3.0, support_m: float = 1.0) -> np.ndarray:  # fmt: skip
    """Ảnh BEV RGBA (alpha 0 = không camera nào thấy mặt đường ở đó). Hàng 0 là y = +range, cột 0 là x = -range: ảnh
    đặt thẳng lên mặt phẳng x-y của LiDAR (nuScenes: x sang phải, y về phía trước).

    - Có `points` (LiDAR của keyframe): ô mặt đường mà tia từ camera tới nó bị một vật cao chắn (xe, tường, cây — điểm
      LiDAR cao hơn mặt đường nằm gần camera hơn) không lấy màu từ camera đó. Đây là phần ảnh bị "kéo nhoè" thành vệt
      toả ra từ xe trong IPM thuần; bỏ đi thì chỉ còn mặt đường thật.
    - `neighbors`: các keyframe lân cận của cùng scene (ảnh, frame, LiDAR). Mặt đường đứng yên nên ô bị che ở frame này
      thường được camera thấy rõ ở frame khác, lúc xe đi ngang qua gần nó. Mỗi lần nhìn thấy được chấm theo độ nét: gần
      camera và gần tâm ảnh nặng hơn hẳn, nên mỗi ô lấy màu chủ yếu từ lần nhìn gần nhất — lấp được vùng bị che, ảnh ở
      xa hết nhoè. Các frame nối với nhau bằng `global_from_lidar` (ego pose).
    - Lỗ nhỏ còn lại được tô từ xung quanh; vùng lớn không ai thấy để trống cho point cloud hiện ra.
    - Lấy mẫu song tuyến. Xa hơn `fade_from` m ảnh mờ dần.
    """
    n = int(round(2 * range_m / res))
    c = (np.arange(n) + 0.5) * res - range_m
    xs, ys = np.meshgrid(c, c[::-1])  # cột: x tăng dần; hàng: y giảm dần
    r = np.hypot(xs.ravel(), ys.ravel())
    inside = np.flatnonzero(r <= range_m)
    gm = ground_model or (fit_ground(points) if points is not None else GroundModel([0.0, 0.0, z0]))
    gx, gy = xs.ravel()[inside], ys.ravel()[inside]
    ground = np.stack([gx, gy, gm.height(gx, gy)])  # điểm mặt đường (hệ LiDAR của frame này) của từng ô

    def cams_of(imgs, fr):
        return [(img, np.asarray(fr.cameras[k].intrinsic, float), np.asarray(fr.cameras[k].cam_from_lidar, float))
                for k, img in imgs.items() if k in fr.cameras and img is not None]  # fmt: skip

    views = [_View(cams_of(images, frame), points, gm if points is not None else None)]
    ref = np.asarray(frame.global_from_lidar, float) if frame.global_from_lidar else None
    for k, (imgs_k, fr_k, pts_k) in enumerate(neighbors or [], 1):
        if ref is None or not fr_k.global_from_lidar:
            continue
        gm_k = fit_ground(pts_k) if pts_k is not None else None
        from_ref = np.linalg.inv(np.asarray(fr_k.global_from_lidar, float)) @ ref
        views.append(_View(cams_of(imgs_k, fr_k), pts_k, gm_k, from_ref, k))
    multi = len(views) > 1
    rgb, seen, sp = _fuse(views, ground, range_m * 1.5, min_depth, (-1.6, 1.6, -4.5, 3.5))
    if points is not None and len(points):
        seen2d = np.zeros(n * n, bool)
        seen2d[inside[seen]] = True
        rgb2d = np.zeros((n * n, 3), np.float32)
        rgb2d[inside] = rgb
        valid = np.zeros(n * n, bool)
        valid[inside] = True
        grid, rgb2d = _clean(seen2d.reshape(n, n), rgb2d.reshape(n, n, 3), valid.reshape(n, n), res, multi,
                             fill_holes)  # fmt: skip
        seen = grid.ravel()[inside]
        rgb = rgb2d.reshape(-1, 3)[inside]
        if support_m > 0 and multi and len(sp) >= 200:
            seen &= _support_mask(sp, gx, gy, np.hypot(gx, gy), support_m)
    out = np.zeros((n * n, 4), np.uint8)
    cells = inside[seen]
    out[cells, :3] = np.clip(rgb[seen], 0, 255).astype(np.uint8)
    rr = r[cells]
    out[cells, 3] = np.where(rr > fade_from, np.clip(255 - (rr - fade_from) / max(range_m - fade_from, 1e-6) * 175, 80,
                                                     255), 255).astype(np.uint8)  # fmt: skip
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


def lidar_uvd_to_ego(lidar: dict | None, intrinsic: np.ndarray, cam_from_ego: np.ndarray) -> np.ndarray | None:
    """Điểm LiDAR đã chiếu lên ảnh (u, v, độ sâu theo trục camera — file `lidar/` của workspace) -> (N, 3) hệ ego."""
    if not lidar or not lidar.get("u"):
        return None
    u, v, d = (np.asarray(lidar[k], float) for k in ("u", "v", "d"))
    cam = np.stack([(u - intrinsic[0, 2]) / intrinsic[0, 0] * d, (v - intrinsic[1, 2]) / intrinsic[1, 1] * d, d], 1)
    ego_from_cam = np.linalg.inv(cam_from_ego)
    return cam @ ego_from_cam[:3, :3].T + ego_from_cam[:3, 3]


def bev_camera(img: np.ndarray, intrinsic: np.ndarray, cam_from_ego: np.ndarray, x_range=BEV2D_X, y_range=BEV2D_Y,
               res: float = 0.1, min_depth: float = 1.0, fade_from: float = 25.0,
               points_ego: np.ndarray | None = None,
               neighbors: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] | None = None,
               fill_holes: float = 3.0, support_m: float = 1.0) -> np.ndarray:  # fmt: skip
    """Ảnh một camera chiếu xuống mặt đường ego, RGBA. Hàng 0 = x xa nhất (phía trước ở trên), cột 0 = y lớn nhất (bên
    trái ở bên trái): nhìn từ trên xuống, đầu xe hướng lên.

    Có `points_ego` (LiDAR, hệ ego): mặt đường lấy theo LiDAR (dốc, nghiêng) thay vì z = 0, và bỏ các ô bị vật cao che
    khuất (cây, xe, tường bị kéo thành vệt dài) như `bev_mosaic`. Không có LiDAR (video tải lên): homography z = 0.

    `neighbors`: ảnh cùng camera ở các keyframe sau / trước, mỗi cái (ảnh, intrinsic, camera <- ego của nó, ego của nó <-
    ego frame này, LiDAR hệ ego của nó). Mặt đường ở xa phía trước được keyframe sau nhìn gần, nên hết nhoè; chỗ bị xe
    trước che được lấp — như `bev_mosaic`."""
    xs = x_range[1] - (np.arange(int(round((x_range[1] - x_range[0]) / res))) + 0.5) * res
    ys = y_range[1] - (np.arange(int(round((y_range[1] - y_range[0]) / res))) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gxr, gyr = gx.ravel(), gy.ravel()

    def ground_of(pts):
        if pts is None or len(pts) < 200:
            return None
        return fit_ground(pts, z_range=(-1.5, 1.5), min_points=200)

    gm = ground_of(points_ego)
    ground = np.stack([gxr, gyr, (gm or GroundModel([0.0, 0.0, 0.0])).height(gxr, gyr)])
    views = [_View([(img, intrinsic, cam_from_ego)], points_ego if gm else None, gm)]
    for k, (img_k, intr_k, cfe_k, from_ref, pts_k) in enumerate(neighbors or [], 1):
        gm_k = ground_of(pts_k)
        views.append(_View([(img_k, intr_k, cfe_k)], pts_k if gm_k else None, gm_k, np.asarray(from_ref, float), k))
    multi = len(views) > 1
    shadow_range = float(np.hypot(x_range[1], max(map(abs, y_range)))) + 5
    # hệ ego: gốc dưới trục sau, x về trước; camera trước thấy nắp capô -> bỏ ô từ đuôi tới ~1.5 m trước đầu xe
    rgb, seen, sp = _fuse(views, ground, shadow_range, min_depth, (-1.2, 6.0, -1.2, 1.2))
    if gm is not None:
        grid, rgb2d = _clean(seen.reshape(gx.shape), rgb.reshape(gx.shape + (3,)), np.ones(gx.shape, bool), res, multi,
                             fill_holes)  # fmt: skip
        seen, rgb = grid.ravel(), rgb2d.reshape(-1, 3)
        if support_m > 0 and multi and len(sp) >= 200:
            seen &= _support_mask(sp, gxr, gyr, np.hypot(gxr, gyr), support_m)
    out = np.zeros((gx.size, 4), np.uint8)
    out[seen, :3] = np.clip(rgb[seen], 0, 255).astype(np.uint8)
    dist = np.hypot(gxr, gyr)
    out[seen, 3] = np.clip(255 - np.maximum(dist[seen] - fade_from, 0) / max(x_range[1] - fade_from, 1e-6) * 175, 80,
                           255)  # fmt: skip
    return out.reshape(gx.shape[0], gx.shape[1], 4)
