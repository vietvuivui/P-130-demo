"""Optical flow giữa hai ảnh camera liền nhau và dịch box theo flow (ý tưởng của Deep Feature Flow, Zhu et al. 2017,
áp dụng ở mức box: tính kỹ ở keyframe, mang kết quả sang ảnh lân cận bằng flow thay vì chạy lại detector hay đoán).

Dùng OpenCV DIS (Kroeger et al. 2016): không cần GPU, không cần huấn luyện; ở nửa độ phân giải (800x450) mất khoảng
0.06 s một cặp trên CPU 2 nhân, rẻ hơn YOLOE-L 1280 khoảng 50 lần.

    field = FlowField.between(img_sweep, img_key)   # dịch chuyển từng điểm ảnh của ảnh sweep sang keyframe
    box_at_key = field.warp_box(box_at_sweep)

Box được dịch theo từng cạnh: cạnh trái lấy trung vị flow ngang ở dải trái bên trong box, cạnh phải ở dải phải, cạnh
trên/dưới tương tự. Cách này theo được cả vật tiến lại gần (box to ra) mà vẫn bền khi mép box lẫn nền. Chỉ lấy vùng
lõi của box để nền phía sau (đứng yên hoặc chạy ngược chiều) ít ảnh hưởng.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Box nhỏ hơn ngần này (pixel ở độ phân giải của flow) chỉ dịch tịnh tiến theo trung vị cả box
_MIN_EDGE_PX = 10


def _cv2():
    import cv2

    return cv2


def load_gray(path: str | Path, scale: float) -> np.ndarray | None:
    if not Path(path).is_file():
        return None
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


@dataclass
class FlowField:
    """Flow (H, W, 2) tính ở độ phân giải `scale` của ảnh gốc; mọi toạ độ vào/ra theo pixel ảnh gốc."""

    field: np.ndarray
    scale: float

    @classmethod
    def between(cls, gray_a: np.ndarray, gray_b: np.ndarray, scale: float) -> FlowField:
        cv2 = _cv2()
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return cls(dis.calc(gray_a, gray_b, None), scale)

    def _median(self, x1: float, y1: float, x2: float, y2: float, axis: int) -> float | None:
        h, w = self.field.shape[:2]
        c1, c2 = int(np.clip(np.floor(x1), 0, w - 1)), int(np.clip(np.ceil(x2), 1, w))
        r1, r2 = int(np.clip(np.floor(y1), 0, h - 1)), int(np.clip(np.ceil(y2), 1, h))
        if c2 <= c1 or r2 <= r1:
            return None
        return float(np.median(self.field[r1:r2, c1:c2, axis]))

    def warp_box(self, box: Sequence[float]) -> list[float]:
        s = self.scale
        x1, y1, x2, y2 = (float(v) * s for v in box)
        w, h = x2 - x1, y2 - y1
        if w < _MIN_EDGE_PX or h < _MIN_EDGE_PX:
            dx = self._median(x1, y1, x2, y2, 0)
            dy = self._median(x1, y1, x2, y2, 1)
            if dx is None or dy is None:
                return [float(v) for v in box]
            return [(x1 + dx) / s, (y1 + dy) / s, (x2 + dx) / s, (y2 + dy) / s]
        cy1, cy2 = y1 + 0.2 * h, y2 - 0.2 * h  # dải giữa theo chiều dọc (cho cạnh trái/phải)
        cx1, cx2 = x1 + 0.2 * w, x2 - 0.2 * w  # dải giữa theo chiều ngang (cho cạnh trên/dưới)
        left = self._median(x1 + 0.1 * w, cy1, x1 + 0.4 * w, cy2, 0)
        right = self._median(x2 - 0.4 * w, cy1, x2 - 0.1 * w, cy2, 0)
        top = self._median(cx1, y1 + 0.1 * h, cx2, y1 + 0.4 * h, 1)
        bottom = self._median(cx1, y2 - 0.4 * h, cx2, y2 - 0.1 * h, 1)
        if None in (left, right, top, bottom):
            return [float(v) for v in box]
        nx1, nx2 = x1 + left, x2 + right
        ny1, ny2 = y1 + top, y2 + bottom
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:  # flow lộn xộn làm box sụp: chỉ tịnh tiến
            dx, dy = (left + right) / 2, (top + bottom) / 2
            nx1, nx2, ny1, ny2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
        return [nx1 / s, ny1 / s, nx2 / s, ny2 / s]


class FlowProvider:
    """Tính (và nhớ ảnh xám vừa đọc) flow giữa hai ảnh theo đường dẫn. resolve(path) -> file ảnh."""

    def __init__(self, resolve: Callable[[str], Path], scale: float = 0.5):
        self.resolve = resolve
        self.scale = scale
        self._gray: dict[str, np.ndarray | None] = {}

    def gray(self, path: str) -> np.ndarray | None:
        if path not in self._gray:
            if len(self._gray) > 8:  # chỉ cần vài ảnh gần nhất
                self._gray.pop(next(iter(self._gray)))
            self._gray[path] = load_gray(self.resolve(path), self.scale)
        return self._gray[path]

    def between(self, path_a: str, path_b: str) -> FlowField | None:
        a, b = self.gray(path_a), self.gray(path_b)
        if a is None or b is None or a.shape != b.shape:
            return None
        return FlowField.between(a, b, self.scale)
