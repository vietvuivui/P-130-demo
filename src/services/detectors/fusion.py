"""Gộp box từ nhiều model (và box trùng lặp của cùng model) thành một object.

Kiểu Weighted Boxes Fusion rút gọn: gom cụm không phân biệt lớp theo IoU,
score của mỗi lớp = tổng score cao nhất của từng model / số model, nên object
chỉ một model thấy tự nhiên bị giảm score.
"""

from __future__ import annotations

import numpy as np

from src.models.schemas import Detection
from src.services.geometry import iou


def fuse_detections(per_model: dict[str, list[Detection]], iou_thr: float) -> list[Detection]:
    n_models = max(1, len(per_model))
    items = [(m, d) for m, dets in per_model.items() for d in dets]
    items.sort(key=lambda x: -x[1].score)

    clusters: list[list[tuple[str, Detection]]] = []
    for model, det in items:
        for cluster in clusters:
            if iou(cluster[0][1].bbox, det.bbox) >= iou_thr:
                cluster.append((model, det))
                break
        else:
            clusters.append([(model, det)])

    fused = []
    for cluster in clusters:
        # label -> model -> score cao nhất
        best: dict[str, dict[str, float]] = {}
        for model, det in cluster:
            by_model = best.setdefault(det.label, {})
            by_model[model] = max(by_model.get(model, 0.0), det.score)
        label_scores = {label: sum(ms.values()) / n_models for label, ms in best.items()}
        label = max(label_scores, key=label_scores.get)

        members = [d for _, d in cluster if d.label == label]
        weights = np.array([d.score for d in members])
        bbox = (np.array([d.bbox for d in members]) * weights[:, None]).sum(0) / weights.sum()

        fused.append(
            Detection(
                bbox=[round(float(v), 1) for v in bbox],
                label=label,
                score=round(label_scores[label], 4),
                models={m: round(s, 4) for m, s in best[label].items()},
                alternatives={lb: round(s, 4) for lb, s in label_scores.items() if lb != label},
            )
        )
    fused.sort(key=lambda d: -d.score)
    return fused
