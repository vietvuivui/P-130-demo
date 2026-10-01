"""Đo thời gian từng bước xử lý, để biết chính xác một frame tốn thời gian ở đâu (detect, LiDAR, flow, QA, kiểm chứng
3D, làm mờ ảnh, nạp model).

- Stopwatch: đo các bước của MỘT frame, ghi vào `FrameRecord.autolabel_timing` / `Frame3DRecord.autolabel_timing`.
- record() / snapshot(): số liệu của cả tiến trình server cho việc không gắn với frame (nạp model, làm mờ ảnh lần đầu).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

# Tên hiển thị trên UI / terminal
STAGE_NAME = {
    "lidar": "Chiếu LiDAR lên ảnh",
    "detect": "Detect 2D (keyframe + sweep)",
    "flow": "Optical flow sweep → keyframe",
    "qa": "QA Agent (risk, cờ)",
    "detect_6cam": "Detect 2D trên 6 camera (kiểm chứng 3D)",
    "points": "Đọc point cloud (kiểm chứng 3D)",
    "verify": "Kiểm chứng box 3D bằng camera",
    "privacy_blur": "Làm mờ mặt / biển số (lần đầu mỗi ảnh)",
    "load_model": "Nạp model detector (một lần)",
}


class Stopwatch:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

    def add(self, name: str, value: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + value

    def result(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.t.items()}


_STATS: dict[str, list[float]] = {}
_LOCK = threading.Lock()


def record(name: str, seconds: float) -> None:
    with _LOCK:
        s = _STATS.setdefault(name, [0, 0.0, 0.0])  # số lần, tổng, lâu nhất
        s[0] += 1
        s[1] += seconds
        s[2] = max(s[2], seconds)


def snapshot() -> dict[str, dict]:
    with _LOCK:
        return {k: {"n": int(v[0]), "total_s": round(v[1], 3), "mean_s": round(v[1] / v[0], 3), "max_s": round(v[2], 3)}
                for k, v in _STATS.items() if v[0]}  # fmt: skip


def summarize(timings: list[dict[str, float]]) -> list[dict]:
    """Gộp timing của nhiều frame: mỗi bước số frame, TB / trung vị giây mỗi frame, tổng, % tổng thời gian."""
    import numpy as np

    keys = [k for k in STAGE_NAME if any(k in t for t in timings)]
    keys += sorted({k for t in timings for k in t} - set(keys) - {"n_model_images"})
    total = sum(sum(v for k, v in t.items() if k != "n_model_images") for t in timings) or 1.0
    rows = []
    for k in keys:
        vals = np.array([t[k] for t in timings if k in t], float)
        if not len(vals):
            continue
        rows.append({"stage": k, "name": STAGE_NAME.get(k, k), "frames": len(vals), "mean_s": round(float(vals.mean()), 3),
                     "median_s": round(float(np.median(vals)), 3), "total_s": round(float(vals.sum()), 2),
                     "share": round(float(vals.sum()) / total, 4)})  # fmt: skip
    return sorted(rows, key=lambda r: -r["total_s"])
