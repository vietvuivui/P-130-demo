"""Dùng box 3D (ensemble LiDAR) làm nguồn nhãn 2D: chiếu box 3D xuống ảnh rồi gộp với box của detector 2D.

Mô hình LiDAR thấy vật xa / tối / bị che tốt hơn detector ảnh, và box chiếu xuống khớp với box 3D mà người duyệt ở
tab 3D. Cách gộp (chọn trên dev 3 scene, đo trên test 24 scene / 957 keyframe CAM_FRONT, eval/results/lidar2d.md):

- box 2D trùng hình chiếu của một box 3D cùng lớp (IoU >= match_iou): giữ box 3D chiếu, điểm = max hai nguồn;
- box 3D không có box 2D trùng: giữ nguyên (vật camera sót);
- box 2D không có box 3D trùng: giữ, điểm x camera_only_scale (LiDAR không thấy thì ít tin hơn).

Test: mAP50 0.366 -> 0.618, precision 0.467 -> 0.671, recall 0.582 -> 0.739 ở ngưỡng giữ box 0.30.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.services.geometry import apply_transform, clip_box, iou, project_points
from src.services.nuscenes_data import box_corners

SOURCE = "lidar3d"  # khoá trong Detection.models của box đến từ mô hình 3D


def project_boxes(preds: Sequence[dict], cam_from_global: np.ndarray, intrinsic: np.ndarray, width: int, height: int,
                  min_score: float = 0.05, min_size_px: float = 2.0) -> list[dict]:  # fmt: skip
    """Box nuScenes (hệ toàn cục) -> [{bbox, label, score}] trên ảnh: hộp bao 8 đỉnh, cắt theo khung ảnh."""
    out = []
    for b in preds:
        score = float(b["detection_score"])
        if score < min_score:
            continue
        cam = apply_transform(cam_from_global, box_corners(b["translation"], b["size"], b["rotation"]))
        front = cam[:, 2] > 0.1
        if not front.any():
            continue
        uv = project_points(cam[front], intrinsic)
        bbox = clip_box([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], width, height)
        if bbox[2] - bbox[0] < min_size_px or bbox[3] - bbox[1] < min_size_px:
            continue
        out.append({"bbox": [round(v, 1) for v in bbox], "label": b["detection_name"], "score": min(score, 1.0)})
    return out


def merge_boxes(boxes3d: Sequence[dict], boxes2d: Sequence[dict], match_iou: float = 0.5,
                camera_only_scale: float = 0.5) -> list[tuple[dict | None, int | None, float]]:  # fmt: skip
    """Ghép tham lam theo điểm box 3D giảm dần. Trả về [(box 3D | None, chỉ số box 2D | None, điểm)]."""
    used: set[int] = set()
    out: list[tuple[dict | None, int | None, float]] = []
    for p in sorted(boxes3d, key=lambda p: -p["score"]):
        best, best_j = match_iou, None
        for j, q in enumerate(boxes2d):
            if j in used or q["label"] != p["label"]:
                continue
            v = iou(p["bbox"], q["bbox"])
            if v >= best:
                best, best_j = v, j
        if best_j is None:
            out.append((p, None, p["score"]))
        else:
            used.add(best_j)
            out.append((p, best_j, max(p["score"], boxes2d[best_j]["score"])))
    out += [(None, j, q["score"] * camera_only_scale) for j, q in enumerate(boxes2d) if j not in used]
    return out


def merge_detections(
    dets: list, boxes3d: Sequence[dict], match_iou: float = 0.5, camera_only_scale: float = 0.5
) -> list:
    """Gộp Detection của detector 2D với box 3D đã chiếu. Box lấy từ 3D bỏ mask (mask của detector không còn khớp)."""
    from src.models.schemas import Detection

    plain = [{"bbox": d.bbox, "label": d.label, "score": d.score} for d in dets]
    out = []
    for p, j, score in merge_boxes(boxes3d, plain, match_iou, camera_only_scale):
        score = round(float(score), 4)
        if p is None:
            d = dets[j]
            out.append(
                d.model_copy(update={"score": score, "det_score": d.det_score if d.det_score is not None else d.score})
            )
        elif j is None:
            out.append(Detection(bbox=p["bbox"], label=p["label"], score=score, models={SOURCE: round(p["score"], 4)}))
        else:
            d = dets[j]
            out.append(Detection(bbox=p["bbox"], label=p["label"], score=score, alternatives=d.alternatives,
                                 models={**d.models, SOURCE: round(p["score"], 4)}))  # fmt: skip
    return sorted(out, key=lambda d: -d.score)
