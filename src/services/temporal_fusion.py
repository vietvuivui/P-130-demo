"""Dùng các sweep lân cận để tính lại score của detection ở keyframe (Flow-Guided Feature Aggregation, Zhu et al. 2017,
làm ở mức box thay vì feature map: không cần huấn luyện lại detector).

Detector đã chạy trên t-2..t+2 cho check temporal, nhưng trước đây kết quả chỉ để gắn cờ. Ở đây box của mỗi sweep được
dịch về thời điểm keyframe bằng optical flow (src/services/flow.py), ghép một-một với box keyframe cùng lớp, rồi:

    mean   : score = (score keyframe + tổng score box ghép được ở từng sweep) / (1 + số sweep có ảnh)
             sweep không thấy vật góp 0 -> box chỉ loé lên ở keyframe bị hạ mạnh; như Weighted Boxes Fusion theo thời
             gian (object một "model" không thấy bị giảm điểm).
    linked : score = trung bình score keyframe và các box ghép được (không phạt sweep không thấy), như Seq-NMS.
    off    : giữ score detector.

Score gốc của detector được giữ ở `det_score`. Người đã sửa box ở sweep (giữ / sửa / vẽ thêm) thì box đó có score 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.models.schemas import Detection
from src.services.geometry import iou_matrix


def match_same_label(
    key: Sequence[tuple[list[float], str]], other: Sequence[tuple[list[float], str]], thr: float
) -> dict[int, int]:
    """Ghép một-một theo IoU giảm dần, chỉ giữa box cùng lớp. {chỉ số trong key: chỉ số trong other}."""
    if not key or not other:
        return {}
    m = iou_matrix(np.array([b for b, _ in key]), np.array([b for b, _ in other]))
    same = np.array([[lk == lo for _, lo in other] for _, lk in key])
    m = np.where(same, m, 0.0)
    rows, cols = np.where(m >= thr)
    out: dict[int, int] = {}
    used: set[int] = set()
    for i, j in sorted(zip(rows.tolist(), cols.tolist(), strict=True), key=lambda ij: -m[ij]):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def temporal_score(s0: float, matched: list[float], n_sweeps: int, mode: str) -> float:
    if mode == "off" or n_sweeps == 0:
        return s0
    if mode == "mean":
        return (s0 + sum(matched)) / (1 + n_sweeps)
    if mode == "linked":
        return (s0 + sum(matched)) / (1 + len(matched))
    raise ValueError(f"temporal.rescore không hợp lệ: {mode}")


def rescore(
    key_dets: list[Detection],
    sweeps: dict[int, list[Detection]],
    warped: dict[int, list[list[float]]],
    mode: str,
    thr: float,
) -> list[Detection]:
    """Detection keyframe với score mới (det_score = score detector). warped[o][j]: box j của sweep o dời về keyframe."""
    if mode == "off" or not sweeps:
        return key_dets
    key = [(d.bbox, d.label) for d in key_dets]
    matched: list[list[float]] = [[] for _ in key_dets]
    for o, dets in sweeps.items():
        boxes = warped.get(o) or [d.bbox for d in dets]
        pairs = match_same_label(key, [(b, d.label) for b, d in zip(boxes, dets, strict=True)], thr)
        for i, j in pairs.items():
            matched[i].append(dets[j].score)
    out = []
    for d, m in zip(key_dets, matched, strict=True):
        base = d.det_score if d.det_score is not None else d.score
        s = temporal_score(base, m, len(sweeps), mode)
        out.append(d.model_copy(update={"score": round(float(min(1.0, s)), 4), "det_score": base}))
    return out
