"""3.2 LiDAR support check: box "ma" và box có kích thước không khớp khoảng cách."""

from __future__ import annotations

import numpy as np

from src.agents.state import QAState
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import QAIssue


def check_lidar(
    bbox: list[float],
    label: str,
    uv: np.ndarray | None,
    depth: np.ndarray | None,
    fy: float,
    image_height: int,
    config: AutoLabelConfig,
) -> dict:
    cfg = config.qa.lidar
    if uv is None or depth is None:
        return {"available": False, "issues": [], "term": 0.0}

    x1, y1, x2, y2 = bbox
    box_h = y2 - y1
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    n_points = int(inside.sum())
    result: dict = {"available": True, "n_points": n_points, "issues": [], "term": 0.0}

    if n_points < cfg.min_points:
        if box_h >= cfg.min_box_height_px:
            result["issues"].append(
                QAIssue(
                    code="NO_LIDAR_SUPPORT",
                    group="lidar",
                    message=f"Chỉ {n_points} điểm LiDAR trong box cao {box_h:.0f}px (cần ≥ {cfg.min_points})",
                )
            )
            result["term"] = 1.0
        else:
            # Box nhỏ ở xa: LiDAR không phủ tới nên không kết luận được, chỉ cộng chút rủi ro
            result["term"] = 0.2
        return result

    # Độ sâu của object: lấy điểm ở vùng giữa box (bớt nền lọt vào) rồi lấy phân vị thấp,
    # vì điểm trên bề mặt object luôn gần hơn nền phía sau
    cx1, cx2 = x1 + 0.25 * (x2 - x1), x2 - 0.25 * (x2 - x1)
    cy1, cy2 = y1 + 0.2 * box_h, y2 - 0.2 * box_h
    central = inside & (uv[:, 0] >= cx1) & (uv[:, 0] <= cx2) & (uv[:, 1] >= cy1) & (uv[:, 1] <= cy2)
    d = depth[central] if central.sum() >= 3 else depth[inside]
    obj_depth = float(np.percentile(d, 25))
    result["depth_m"] = round(obj_depth, 2)

    # Box chạm mép trên/dưới thì chiều cao bị cắt, không so kích thước được
    truncated = y1 <= 2 or y2 >= image_height - 2
    spec = config.classes.get(label)
    if spec is None or truncated or obj_depth > cfg.max_depth_m:
        return result

    est_h = box_h * obj_depth / fy
    result["est_height_m"] = round(est_h, 2)
    lo, hi = spec.height_m
    a, b = cfg.size_tolerance
    if est_h < lo * a or est_h > hi * b:
        result["issues"].append(
            QAIssue(
                code="SIZE_DEPTH_MISMATCH",
                group="lidar",
                message=f"Cao ước lượng {est_h:.1f} m ở độ sâu {obj_depth:.1f} m, lớp {label} hợp lý {lo}–{hi} m",
            )
        )
        result["term"] = 0.6
    return result


def lidar_node(state: QAState) -> dict:
    config = state["config"]
    fy = state["intrinsic"][1][1]
    height = state["image_size"][1]
    uv, depth = state.get("lidar_uv"), state.get("lidar_depth")
    return {
        "lidar": {
            o.object_id: check_lidar(o.bbox, o.label, uv, depth, fy, height, config) for o in state.get("objects", [])
        }
    }
