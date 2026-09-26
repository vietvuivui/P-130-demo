"""Hàm hình học dùng chung: IoU box 2D, quaternion, phép biến đổi 4x4."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp box, a: (N, 4), b: (M, 4) -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def clip_box(b: Sequence[float], width: float, height: float) -> list[float]:
    return [
        float(np.clip(b[0], 0, width)),
        float(np.clip(b[1], 0, height)),
        float(np.clip(b[2], 0, width)),
        float(np.clip(b[3], 0, height)),
    ]


def interpolate_box(b0: Sequence[float], t0: float, b1: Sequence[float], t1: float, t: float) -> list[float]:
    """Nội suy tuyến tính box theo thời gian."""
    if t1 == t0:
        return [float(v) for v in b0]
    w = (t - t0) / (t1 - t0)
    return [float(x0 + w * (x1 - x0)) for x0, x1 in zip(b0, b1, strict=True)]


def touches_border(b: Sequence[float], width: float, height: float, margin: float) -> bool:
    return b[0] <= margin or b[1] <= margin or b[2] >= width - margin or b[3] >= height - margin


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Quaternion (w, x, y, z) như nuScenes -> ma trận xoay 3x3."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_matrix(translation: Sequence[float], rotation: Sequence[float], inverse: bool = False) -> np.ndarray:
    """Ma trận 4x4 từ translation + quaternion. inverse=True cho phép biến đổi ngược."""
    r = quaternion_to_matrix(rotation)
    t = np.asarray(translation, dtype=np.float64)
    m = np.eye(4)
    if inverse:
        m[:3, :3] = r.T
        m[:3, 3] = -r.T @ t
    else:
        m[:3, :3] = r
        m[:3, 3] = t
    return m


def apply_transform(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Áp ma trận 4x4 lên điểm (N, 3)."""
    return points @ m[:3, :3].T + m[:3, 3]


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Chiếu điểm trong hệ camera (N, 3) xuống ảnh -> (N, 2)."""
    uvw = points_cam @ np.asarray(intrinsic).T
    return uvw[:, :2] / uvw[:, 2:3]
