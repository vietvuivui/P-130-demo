"""Kiểm chứng từng box 3D bằng camera + LiDAR, không dùng nhãn gốc (QA Agent cho phần 3D).

Chuyển từ bộ kiểm chứng của nhóm 3D (scripts/verify_objects.py, notebook demo/model.ipynb) sang dữ liệu của sản
phẩm: đọc bằng NuScenesMini thay cho nuscenes-devkit, detection 2D lấy từ DetectorEnsemble (YOLOE, có cache).
Giữ nguyên ngưỡng và tên kết luận. Bỏ bước lọc "bãi đỗ xe đạp" vì bước đó cần nhãn gốc.

Với mỗi box 3D:
  1. Chiếu lên 6 camera (cắt theo mặt phẳng gần để góc nằm sau camera không chiếu sai).
  2. Điểm thông tin mỗi camera = phần nằm trong ảnh x (1 - mức che) x độ rõ; chọn camera điểm cao nhất.
  3. Ghép với box 2D của detector (Hungarian; IoU >= 0.4, hoặc khớp một phần khi bị che / cắt mép mà độ sâu LiDAR khớp).
  4. Kết luận -> mức rủi ro để người duyệt theo ngoại lệ như phần 2D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.schemas3d import Verify3D

NEAR = 0.1  # mặt phẳng gần trước camera (m)
MIN_AREA = 100  # box chiếu nhỏ hơn (px^2) thì không xét camera đó
SIZE_REF, SHARP_REF, LIGHT_REF = 60.0, 60.0, 60.0
CLEAR_OCC, CLEAR_INFO = 0.3, 0.5  # "ảnh rõ": che < 30% và điểm thông tin >= 0.5 mới dám kết luận camera không thấy
OCC_MARGIN, MIN_LIDAR_PTS, DUP_DIST = 1.0, 5, 1.5
RIDER_IOA, IOU_THR, IOA_THR, COVER_THR = 0.3, 0.4, 0.7, 0.4
NEAR_Z, DEPTH_MARGIN, SHIFT_THR, MIN_PTS_IN_BOX = 8.0, 1.5, 0.35, 3
CLASS_HEIGHT = {
    "car": 1.6, "truck": 2.8, "bus": 3.2, "trailer": 3.5, "construction_vehicle": 3.0,
    "pedestrian": 1.7, "bicycle": 1.3, "motorcycle": 1.3, "traffic_cone": 0.7, "barrier": 1.0,
}  # fmt: skip
RIDER_HEIGHT = 1.7
# Phạm vi chấm điểm theo lớp của nuScenes (m): box xa hơn không được kiểm chứng, giống bộ chấm
CLASS_RANGE = {
    "car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50,
    "pedestrian": 40, "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30,
}  # fmt: skip
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

V_OK, V_SHIFT, V_CLASS = "DUNG", "DUNG VAT, BOX LECH", "SAI LOP"
V_FP, V_MISS, V_UNSURE = "NGHI BAO NHAM", "CAMERA KHONG XAC NHAN", "CHUA DU THONG TIN"
VERDICT_LEVEL = {V_OK: "low", V_SHIFT: "medium", V_MISS: "medium", V_UNSURE: "medium", V_CLASS: "high", V_FP: "high"}


@dataclass
class Box:
    label: str
    score: float
    center: np.ndarray  # hệ LiDAR
    size: np.ndarray  # w, l, h
    yaw: float

    def corners(self) -> np.ndarray:
        """8 đỉnh (8, 3) theo thứ tự của nuScenes: 0-3 mặt trước (x+), 4-7 mặt sau."""
        w, length, h = self.size
        x = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return (rot @ np.vstack([x, y, z])).T + self.center


def points_in_box(box: Box, pts: np.ndarray) -> int:
    """Số điểm LiDAR (hệ LiDAR) trong box, bỏ lớp sát đáy box (mặt đường): 20 cm hoặc 15% chiều cao."""
    if len(pts) == 0:
        return 0
    c, s = np.cos(-box.yaw), np.sin(-box.yaw)
    d = pts[:, :3] - box.center
    lx, ly, lz = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]
    w, length, h = box.size
    inside = (np.abs(lx) <= length / 2) & (np.abs(ly) <= w / 2) & (lz <= h / 2) & (lz >= -h / 2 + min(0.2, 0.15 * h))
    return int(inside.sum())


def _area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _inter(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _ioa(small, big) -> float:
    return _inter(small, big) / max(_area(small), 1e-9)


def _iou_matrix(a, b) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Bao lồi 2D (thuật toán chuỗi đơn điệu), không cần OpenCV."""
    p = sorted(map(tuple, pts))
    if len(p) <= 2:
        return np.array(p)

    def half(seq):
        h = []
        for q in seq:
            while (
                len(h) >= 2
                and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0
            ):
                h.pop()
            h.append(q)
        return h

    lower, upper = half(p), half(reversed(p))
    return np.array(lower[:-1] + upper[:-1])


def _in_hull(hull: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Điểm nằm trong đa giác lồi (đỉnh ngược chiều kim đồng hồ)."""
    if len(pts) == 0 or len(hull) < 3:
        return np.zeros(len(pts), bool)
    inside = np.ones(len(pts), bool)
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= 0
    return inside


def project_box(box: Box, cam_from_lidar: np.ndarray, intrinsic: np.ndarray, w: int, h: int) -> dict | None:
    """Box -> thông tin 2D trên một camera, hoặc None nếu không lọt vào ảnh."""
    k = (cam_from_lidar[:3, :3] @ box.corners().T) + cam_from_lidar[:3, 3:4]  # (3, 8)
    front = k[2] > NEAR
    if not front.any():
        return None
    pts = [k[:, i] for i in range(8) if front[i]]
    for i, j in EDGES:
        if front[i] != front[j]:
            t = (NEAR - k[2, i]) / (k[2, j] - k[2, i])
            pts.append(k[:, i] + t * (k[:, j] - k[:, i]))
    pts = np.array(pts).T
    uv = (intrinsic @ pts)[:2] / pts[2]
    x1, y1 = uv.min(axis=1)
    x2, y2 = uv.max(axis=1)
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, w), min(y2, h)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    area = (cx2 - cx1) * (cy2 - cy1)
    center = cam_from_lidar[:3, :3] @ box.center + cam_from_lidar[:3, 3]
    return {
        "bbox": [float(cx1), float(cy1), float(cx2), float(cy2)],
        "vis": float(area / max((x2 - x1) * (y2 - y1), 1e-9)),
        "area": float(area),
        "depth": float(center[2]),
        "zmin": float(pts[2].min()),
        "zmax": float(pts[2].max()),
        "hull": _convex_hull(uv.T),
    }


def add_occlusion(views: dict[int, dict], boxes: list[Box], uvz: tuple) -> None:
    """Mức che (0..1) của mỗi box trên một camera.

    Vùng box có đủ điểm LiDAR: tỉ lệ điểm gần camera hơn mặt trước box quá 1 m (vật chắn trên đường nhìn).
    Ít điểm: phần box bị box 3D khác (gần hơn >= 1 m) phủ lên, tính gần đúng bằng lưới 8 px."""
    u, v, z = uvz
    for i, view in views.items():
        x1, y1, x2, y2 = view["bbox"]
        sel = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        zin = z[sel][_in_hull(view["hull"], np.c_[u[sel], v[sel]])] if sel.any() else np.zeros(0)
        if len(zin) >= MIN_LIDAR_PTS:
            view.update(occ=float((zin < view["zmin"] - OCC_MARGIN).mean()), occ_src="lidar")
            continue
        gx, gy = np.meshgrid(np.arange(x1, x2, 8.0) + 4, np.arange(y1, y2, 8.0) + 4)
        grid = np.c_[gx.ravel(), gy.ravel()]
        own = _in_hull(view["hull"], grid)
        covered = np.zeros(len(grid), bool)
        for j, other in views.items():
            if (
                j != i
                and other["depth"] < view["depth"] - OCC_MARGIN
                and np.linalg.norm(boxes[j].center - boxes[i].center) >= DUP_DIST
            ):
                covered |= _in_hull(other["hull"], grid)
        n_own = own.sum()
        view.update(occ=float((own & covered).sum() / n_own) if n_own else 0.0, occ_src="box")


def add_clarity(views: dict[int, dict], gray: np.ndarray | None) -> None:
    """Độ rõ = kích thước x độ nét (phương sai Laplacian) x độ sáng của vùng box."""
    for view in views.values():
        size = min(1.0, np.sqrt(view["area"]) / SIZE_REF)
        sharp = light = 60.0
        if gray is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in view["bbox"])
            crop = gray[y1 : max(y2, y1 + 3), x1 : max(x2, x1 + 3)].astype(np.float64)
            if crop.size:
                lap = crop[1:-1, 1:-1] * -4 + crop[:-2, 1:-1] + crop[2:, 1:-1] + crop[1:-1, :-2] + crop[1:-1, 2:]
                sharp = float(lap.var()) if lap.size else 0.0
                light = float(crop.mean())
        view["clarity"] = size * max(0.3, min(1.0, sharp / SHARP_REF)) * max(0.3, min(1.0, light / LIGHT_REF))
        view["info"] = view["vis"] * (1.0 - view["occ"]) * view["clarity"]


def merge_riders(dets: list[dict]) -> list[dict]:
    """nuScenes: box xe đạp / xe máy bao cả người lái -> gộp box người ngồi trên xe vào box xe."""
    drop = set()
    for i, p in enumerate(dets):
        if p["label"] != "pedestrian":
            continue
        cx = (p["bbox"][0] + p["bbox"][2]) / 2
        cands = [
            (_ioa(p["bbox"], c["bbox"]), c)
            for c in dets
            if c["label"] in ("bicycle", "motorcycle") and not c.get("rider") and c["bbox"][0] <= cx <= c["bbox"][2]
        ]
        ioa, cyc = max(cands, key=lambda t: t[0], default=(0.0, None))
        if cyc is not None and ioa >= RIDER_IOA:
            a, b = cyc["bbox"], p["bbox"]
            cyc.update(
                bbox=[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                rider=True,
                score=max(cyc["score"], p["score"]),
            )
            drop.add(i)
    return [d for i, d in enumerate(dets) if i not in drop]


def det_depth(d: dict, uvz: tuple, intrinsic: np.ndarray, h: int) -> tuple[float, float]:
    """(độ sâu, sai số) của box 2D: phân vị 30% độ sâu LiDAR trong box; thiếu điểm thì ước từ chiều cao box."""
    u, v, z = uvz
    x1, y1, x2, y2 = d["bbox"]
    sel = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if sel.sum() >= 3:
        return float(np.percentile(z[sel], 30)), DEPTH_MARGIN
    if y1 <= 1 or y2 >= h - 1:
        return np.nan, np.nan
    zz = intrinsic[1, 1] * (RIDER_HEIGHT if d.get("rider") else CLASS_HEIGHT.get(d["label"], 1.5)) / max(y2 - y1, 1.0)
    return zz, 0.3 * zz


def match_camera(views: dict[int, dict], boxes: list[Box], dets: list[dict], depths: list) -> dict[int, tuple]:
    """Hungarian trên một camera: {chỉ số box: (chỉ số box 2D, IoU, kiểu khớp)}."""
    from scipy.optimize import linear_sum_assignment

    idx = list(views)
    if not idx or not dets:
        return {}
    iou = _iou_matrix([views[i]["bbox"] for i in idx], [d["bbox"] for d in dets])
    score = np.where(iou >= IOU_THR, iou, 0.0)
    for a, i in enumerate(idx):
        v = views[i]
        partial = v["occ"] >= 0.3 or v["vis"] < 0.95 or v["zmin"] < NEAR_Z
        visible_area = _area(v["bbox"]) * max(1.0 - v["occ"], 0.2)
        for j, d in enumerate(dets):
            z, margin = depths[j]
            depth_ok = not np.isnan(z) and v["zmin"] - margin <= z <= v["zmax"] + margin
            if score[a, j] > 0:
                if boxes[i].label != d["label"] and not np.isnan(z) and not depth_ok:
                    score[a, j] = 0.0
                continue
            if (
                partial
                and depth_ok
                and _ioa(d["bbox"], v["bbox"]) >= IOA_THR
                and _inter(d["bbox"], v["bbox"]) / visible_area >= COVER_THR
            ):
                score[a, j] = 0.01 + 0.01 * _ioa(d["bbox"], v["bbox"])
    same = np.array([[boxes[i].label == d["label"] for d in dets] for i in idx])
    cost = np.where(score > 0, -(score + 0.05 * same), 1e6)
    out = {}
    for a, j in zip(*linear_sum_assignment(cost), strict=True):
        if score[a, j] > 0:
            out[idx[a]] = (int(j), float(iou[a, j]), "IoU" if iou[a, j] >= IOU_THR else "mot phan")
    return out


def _shift(v: dict, d: dict) -> tuple[float, float]:
    (a1, _b1, a2, b2), (c1, d1, c2, d2) = v["bbox"], d["bbox"]
    w, h = max(c2 - c1, 1.0), max(d2 - d1, 1.0)
    return abs((a1 + a2) / 2 - (c1 + c2) / 2) / w, abs(b2 - d2) / h


def conclude(box: Box, best_cam, v, match, d, mv, n_pts: int) -> tuple[str, str]:
    """Kết luận + một dòng số liệu cho thẻ box. Ý nghĩa của kết luận đã nằm ở nhãn (verdict) trên thẻ, nên dòng này chỉ
    ghi số: IoU, độ lệch, camera, mức thấy / che, số điểm LiDAR."""
    pts_txt = f"{n_pts} điểm LiDAR"
    if best_cam is None:
        return V_UNSURE, f"Không lọt rõ vào camera nào · {pts_txt}"
    view_txt = f"{best_cam} · thấy {v['vis']:.0%} · che {v['occ']:.0%} · độ rõ {v['clarity']:.2f}"
    if match is not None:
        cam, (_j, iou, kind) = match
        where = "" if cam == best_cam else f" · xác nhận ở {cam}"
        said = (
            d["label"]
            if d.get("prompt", d["label"]).replace(" ", "_") == d["label"]
            else f"{d['label']} ('{d['prompt']}')"
        )
        if d["label"] != box.label:
            return V_CLASS, f"Camera: {said} {d['score']:.2f} · 3D: {box.label}{where} · {pts_txt}"
        if kind == "mot phan":
            return V_OK, f"Khớp một phần (bị che / cắt mép){where} · {pts_txt}"
        dx, dy = _shift(mv, d)
        if iou < 0.5 and (dx > SHIFT_THR or dy > SHIFT_THR):
            return V_SHIFT, f"IoU {iou:.2f} · tâm lệch {dx:.0%} · đáy lệch {dy:.0%}{where} · {pts_txt}"
        return V_OK, f"IoU {iou:.2f}{where} · {pts_txt}"
    if v["occ"] >= CLEAR_OCC or v["info"] < CLEAR_INFO:
        return V_UNSURE, f"{view_txt} · 2D không thấy · {pts_txt}"
    return (V_FP if n_pts < MIN_PTS_IN_BOX else V_MISS), f"{view_txt} · 2D không thấy · {pts_txt}"


@dataclass
class CamInput:
    name: str
    K: np.ndarray
    cam_from_lidar: np.ndarray
    width: int
    height: int
    gray: np.ndarray | None
    dets: list[dict]  # {"bbox", "label", "score", "prompt"}


def verify_boxes(boxes: list[Box], cams: list[CamInput], lidar_pts: np.ndarray) -> list[Verify3D]:
    """Kiểm chứng mọi box 3D của một keyframe. lidar_pts: (N, >=3) trong hệ LiDAR của keyframe."""
    views: dict[str, dict[int, dict]] = {}
    uvz: dict[str, tuple] = {}
    matches: dict[int, dict[str, tuple]] = {i: {} for i in range(len(boxes))}
    dets_by_cam: dict[str, list[dict]] = {}
    for cam in cams:
        p = (cam.cam_from_lidar[:3, :3] @ lidar_pts[:, :3].T) + cam.cam_from_lidar[:3, 3:4]
        p = p[:, p[2] > NEAR]
        uv = (cam.K @ p)[:2] / p[2]
        ok = (uv[0] >= 0) & (uv[0] < cam.width) & (uv[1] >= 0) & (uv[1] < cam.height)
        uvz[cam.name] = (uv[0, ok], uv[1, ok], p[2, ok])
        vs = {}
        for i, b in enumerate(boxes):
            v = project_box(b, cam.cam_from_lidar, cam.K, cam.width, cam.height)
            if v is not None and v["area"] >= MIN_AREA:
                vs[i] = v
        add_occlusion(vs, boxes, uvz[cam.name])
        add_clarity(vs, cam.gray)
        views[cam.name] = vs
        dets = merge_riders([dict(d) for d in cam.dets])
        dets_by_cam[cam.name] = dets
        depths = [det_depth(d, uvz[cam.name], cam.K, cam.height) for d in dets]
        for i, m in match_camera(vs, boxes, dets, depths).items():
            matches[i][cam.name] = m

    out = []
    for i, b in enumerate(boxes):
        ranked = sorted(((views[c.name][i]["info"], c.name) for c in cams if i in views[c.name]), reverse=True)
        best = ranked[0][1] if ranked else None
        v = views[best][i] if best else None
        match = None
        if best in matches[i]:
            match = (best, matches[i][best])
        elif matches[i]:
            c = max(matches[i], key=lambda c: views[c][i]["info"])
            match = (c, matches[i][c])
        d = dets_by_cam[match[0]][match[1][0]] if match else None
        mv = views[match[0]][i] if match else None
        n_pts = points_in_box(b, lidar_pts)
        verdict, comment = conclude(b, best, v, match, d, mv, n_pts)
        out.append(
            Verify3D(
                verdict=verdict,
                level=VERDICT_LEVEL[verdict],
                comment=comment,
                camera=best,
                cameras={c: round(s, 3) for s, c in ranked},
                bbox2d={c: [round(x, 1) for x in views[c][i]["bbox"]] for c in views if i in views[c]},
                visible=round(v["vis"], 3) if v else None,
                occlusion=round(v["occ"], 3) if v else None,
                clarity=round(v["clarity"], 3) if v else None,
                visible_by_cam={c: round(views[c][i]["vis"], 3) for c in views if i in views[c]},
                occlusion_by_cam={c: round(views[c][i]["occ"], 3) for c in views if i in views[c]},
                lidar_points=n_pts,
                distance_m=round(float(np.hypot(b.center[0], b.center[1])), 1),
                det_label=d["label"] if d else None,
                det_prompt=d.get("prompt") if d else None,
                det_score=round(d["score"], 3) if d else None,
                det_bbox=[round(x, 1) for x in d["bbox"]] if d else None,
                det_camera=match[0] if match else None,
                iou=round(match[1][1], 3) if match else None,
            )
        )
    return out
