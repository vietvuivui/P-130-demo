"""Đề xuất box 3D cho vật camera thấy mà mô hình 3D bỏ sót (ý tưởng của VESPA, arXiv:2507.20397, rút gọn).

Với mỗi box 2D của detector (6 camera) không trùng hình chiếu của box 3D nào:

1. lấy điểm LiDAR (gộp vài lần quét) cao hơn mặt đường (`bev.fit_ground`) rơi vào box 2D (hoặc mask nếu có);
2. gom cụm trên mặt phẳng x-y (DBSCAN), bỏ cụm to hơn hẳn cỡ của lớp (tường, hàng rào phía sau vật), lấy cụm nhiều
   điểm nhất;
3. dựng box: hướng theo hình chữ nhật nhỏ nhất bao cụm (vật có hướng) hoặc vuông góc tia nhìn; kích thước = cỡ trung
   bình của lớp nhưng không nhỏ hơn cụm; tâm dời ra xa cảm biến khi LiDAR chỉ thấy một mặt của vật; đáy đặt lên mặt đường;
4. điểm = điểm 2D × `score_scale`, để box đề xuất xếp sau box của mô hình 3D và luôn qua người duyệt.

Không cần mô hình mới, không cần GPU ngoài detector 2D đã chạy cho bước kiểm chứng 3D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.services.bev import GroundModel

# Cỡ trung bình (rộng, dài, cao) theo lớp trên nuScenes train
CLASS_SIZE = {
    "car": (1.95, 4.62, 1.73), "truck": (2.51, 6.93, 2.84), "bus": (2.94, 11.19, 3.47),
    "trailer": (2.90, 12.28, 3.87), "construction_vehicle": (2.82, 6.37, 3.19), "pedestrian": (0.67, 0.73, 1.77),
    "motorcycle": (0.77, 2.11, 1.47), "bicycle": (0.61, 1.70, 1.29), "traffic_cone": (0.41, 0.41, 1.07),
    "barrier": (2.53, 0.50, 0.98),
}  # fmt: skip
ORIENTED = {"car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle", "bicycle", "barrier"}
ATTRIBUTE = {
    "car": "vehicle.parked", "truck": "vehicle.parked", "bus": "vehicle.stopped", "trailer": "vehicle.parked",
    "construction_vehicle": "vehicle.parked", "pedestrian": "pedestrian.standing",
    "motorcycle": "cycle.without_rider", "bicycle": "cycle.without_rider", "traffic_cone": "", "barrier": "",
}  # fmt: skip
CLASS_RANGE = {
    "car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50,
    "pedestrian": 40, "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30,
}  # fmt: skip


@dataclass
class FillCfg:
    # Dev (3 scene): đề xuất cho xe / barrier gần như luôn sai (~5% đúng — cụm điểm là xe đã có box lệch, tường, bụi
    # cây); người, cọc tiêu, xe đạp đúng 50–75%. Mặc định chỉ đề xuất các lớp nhỏ này.
    classes: frozenset = frozenset({"pedestrian", "traffic_cone", "bicycle", "motorcycle"})
    min_det_score: float = 0.6  # box 2D yếu hơn không đề xuất
    explained_iou: float = 0.25  # box 2D trùng hình chiếu box 3D sẵn có từ ngưỡng này: không phải vật bị sót
    min_points: dict[str, int] = field(default_factory=lambda: {"pedestrian": 8, "traffic_cone": 6, "bicycle": 8,
                                                                "motorcycle": 10, "barrier": 10})  # fmt: skip
    default_min_points: int = 16
    shrink: float = 0.15  # bỏ viền mỗi bên của box 2D (điểm nền lọt vào rìa)
    min_height: float = 0.2
    score_scale: float = 0.6
    near_existing_m: float = 1.0  # tâm đề xuất gần box 3D sẵn có hơn: bỏ (đã có vật ở đó)
    use_mask: bool = True


@dataclass
class Cam:
    name: str
    intrinsic: np.ndarray
    cam_from_lidar: np.ndarray
    width: int
    height: int
    dets: list  # Detection (bbox, label, score, mask)


@dataclass
class Proposal:
    label: str
    score: float
    center: np.ndarray  # hệ LiDAR
    size: np.ndarray  # w, l, h
    yaw: float
    n_points: int
    camera: str
    det_score: float


def box_corners(center, size, yaw) -> np.ndarray:
    w, length, h = size
    x = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * x - s * y, s * x + c * y, z], 1) + np.asarray(center)[:3]


def project_bbox(corners: np.ndarray, cam: Cam) -> list[float] | None:
    pc = corners @ cam.cam_from_lidar[:3, :3].T + cam.cam_from_lidar[:3, 3]
    if (pc[:, 2] > 0.1).sum() < 4:
        return None
    pc = pc[pc[:, 2] > 0.1]
    uv = pc @ cam.intrinsic.T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    x1, y1, x2, y2 = max(u.min(), 0), max(v.min(), 0), min(u.max(), cam.width), min(v.max(), cam.height)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def iou2d(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _in_mask(u: np.ndarray, v: np.ndarray, poly: list[float] | None, bbox: list[float], shrink: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink * 0.5
    inside = (u >= x1 + dx) & (u <= x2 - dx) & (v >= y1 + dy) & (v <= y2 - dy)
    if poly and len(poly) >= 6 and inside.any():
        import cv2

        pts = np.round(np.asarray(poly, float).reshape(-1, 2) - [x1, y1]).astype(np.int32)
        mw, mh = int(np.ceil(x2 - x1)) + 2, int(np.ceil(y2 - y1)) + 2
        mask = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        idx = np.flatnonzero(inside)
        cu = np.clip((u[idx] - x1).astype(int), 0, mw - 1)
        cv = np.clip((v[idx] - y1).astype(int), 0, mh - 1)
        inside[idx] = mask[cv, cu] > 0
    return inside


def _cluster(xy: np.ndarray, eps: float) -> np.ndarray:
    """Gom cụm liên kết đơn: hai điểm cách nhau <= eps chung cụm (như DBSCAN min_samples=2); điểm lẻ nhận -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(xy).query_pairs(eps, output_type="ndarray")
    n = len(xy)
    graph = (
        coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        if len(pairs)
        else coo_matrix((n, n))
    )
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return np.where(sizes[labels] >= 2, labels, -1)


def fit_box(pts: np.ndarray, label: str, ground: GroundModel) -> tuple[np.ndarray, np.ndarray, float]:
    """Box từ cụm điểm (N, 3) hệ LiDAR, cảm biến ở gốc toạ độ."""
    w0, l0, h0 = CLASS_SIZE[label]
    xy = pts[:, :2]
    centroid = xy.mean(0)
    ray = math.atan2(centroid[1], centroid[0])
    if label in ORIENTED and len(xy) >= 5:
        import cv2

        (_, _), (ew, eh), ang = cv2.minAreaRect(xy.astype(np.float32))
        yaw = math.radians(ang) if ew >= eh else math.radians(ang + 90)
        if max(ew, eh) < 0.6 * l0 and max(ew, eh) < 1.0:  # cụm quá nhỏ để tin hướng: dọc theo tia nhìn
            yaw = ray
    else:
        yaw = ray
    c, s = math.cos(yaw), math.sin(yaw)
    rel = xy - centroid
    along, across = rel @ np.array([c, s]), rel @ np.array([-s, c])
    ext_l, ext_w = float(np.ptp(along)), float(np.ptp(across))
    length, width = max(l0, ext_l), max(w0, ext_w)
    mid = (
        centroid
        + np.array([c, s]) * (along.max() + along.min()) / 2
        + np.array([-s, c]) * (across.max() + across.min()) / 2
    )
    # LiDAR chỉ thấy mặt gần: dời tâm ra xa cảm biến theo mỗi trục phần còn thiếu của cỡ lớp
    away = mid / (np.linalg.norm(mid) + 1e-9)
    for axis, ext, full in ((np.array([c, s]), ext_l, length), (np.array([-s, c]), ext_w, width)):
        mid = mid + axis * np.sign(axis @ away) * max(full - ext, 0.0) / 2
    zg = float(ground.height(mid[:1], mid[1:2])[0])
    height = max(h0, float(pts[:, 2].max() - zg))
    return np.array([mid[0], mid[1], zg + height / 2]), np.array([width, length, height]), yaw


def propose(points: np.ndarray, ground: GroundModel, cams: list[Cam], existing: list[tuple[str, np.ndarray,
            np.ndarray, float]], cfg: FillCfg | None = None) -> list[Proposal]:  # fmt: skip
    """points: (N, >=3) hệ LiDAR keyframe (đã gộp sweep). existing: (lớp, tâm, cỡ, yaw) box 3D sẵn có, hệ LiDAR."""
    cfg = cfg or FillCfg()
    hgt = points[:, 2] - ground.height(points[:, 0], points[:, 1])
    obs = points[(hgt > cfg.min_height) & (hgt < 4.5), :3]
    ex_corners = [box_corners(c, s, y) for _, c, s, y in existing]
    ex_xy = np.array([c[:2] for _, c, _, _ in existing]) if existing else np.zeros((0, 2))
    out: list[Proposal] = []
    for cam in cams:
        proj = [project_bbox(k, cam) for k in ex_corners]
        pc = obs @ cam.cam_from_lidar[:3, :3].T + cam.cam_from_lidar[:3, 3]
        front = pc[:, 2] > 0.5
        uv = pc[front] @ cam.intrinsic.T
        u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        cand = obs[front]
        for d in cam.dets:
            label = d.label
            if label not in CLASS_SIZE or label not in cfg.classes or d.score < cfg.min_det_score:
                continue
            if any(p is not None and iou2d(d.bbox, p) >= cfg.explained_iou for p in proj):
                continue
            sel = _in_mask(u, v, d.mask if cfg.use_mask else None, d.bbox, cfg.shrink)
            need = cfg.min_points.get(label, cfg.default_min_points)
            if sel.sum() < need:
                continue
            p = cand[sel]
            w0, l0, _ = CLASS_SIZE[label]
            labels = _cluster(p[:, :2], 0.7 if label in ("car", "truck", "bus", "trailer", "construction_vehicle")
                              else 0.4)  # fmt: skip
            best, best_n = None, 0
            for k in set(labels) - {-1}:
                m = labels == k
                span = float(np.ptp(p[m, 0])) + float(np.ptp(p[m, 1]))
                if span > (l0 + w0) * 1.6 + 0.5:  # to hơn hẳn vật: nền phía sau (tường, hàng rào, bụi cây)
                    continue
                if m.sum() > best_n:
                    best, best_n = m, int(m.sum())
            if best is None or best_n < need:
                continue
            center, size, yaw = fit_box(p[best], label, ground)
            if math.hypot(center[0], center[1]) > CLASS_RANGE[label]:
                continue
            if len(ex_xy) and np.min(np.hypot(*(ex_xy - center[:2]).T)) < cfg.near_existing_m:
                continue
            out.append(Proposal(label, round(d.score * cfg.score_scale, 4), center, size, yaw, best_n, cam.name,
                                d.score))  # fmt: skip
    # cùng vật thấy ở hai camera (vùng chồng nhau): giữ đề xuất điểm cao hơn
    out.sort(key=lambda q: -q.score)
    kept: list[Proposal] = []
    for q in out:
        if all(math.hypot(*(q.center[:2] - k.center[:2])) > max(0.8, min(CLASS_SIZE[q.label][:2])) for k in kept):
            kept.append(q)
    return kept
