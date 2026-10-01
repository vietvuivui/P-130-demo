from __future__ import annotations

from typing import TypedDict

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection, LabelObject


class SweepDetections(TypedDict, total=False):
    timestamp: int
    detections: list[Detection]
    # Box của detections dời về thời điểm keyframe bằng optical flow (cùng thứ tự); thiếu = so khớp trực tiếp
    warped: list[list[float]]


class QAState(TypedDict, total=False):
    """State của QA Agent cho một frame.

    Ba node kiểm tra (confidence / lidar / temporal) chạy song song, mỗi node
    ghi vào key riêng, nên không cần reducer. Node issue_generation gộp lại.
    """

    # Đầu vào
    config: AutoLabelConfig
    image_size: tuple[int, int]  # (width, height)
    intrinsic: list[list[float]]
    key_timestamp: int
    objects: list[LabelObject]  # detection keyframe, đã có object_id
    sweeps: dict[int, SweepDetections]  # offset -> detection ở sweep đó
    lidar_uv: np.ndarray | None
    lidar_depth: np.ndarray | None

    # 3.1 / 3.2 / 3.3: object_id -> {"issues": [...], "term": float, ...}
    confidence: dict[str, dict]
    lidar: dict[str, dict]
    temporal: dict[str, dict]
    recovered: list[LabelObject]  # box đề xuất từ tracking (RECOVERED_BY_TRACK)

    # 3.4 / 3.5
    reviewed: list[LabelObject]  # object kèm issue + risk
    frame_risk: float
