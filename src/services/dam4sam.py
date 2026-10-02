"""DAM4SAM (Videnović et al. 2024): SAM 2.1 với bộ nhớ nhận biết vật gây nhiễu, dùng để lan truyền box của từng vật
từ keyframe đã duyệt sang các ảnh sau — thay cho optical flow + ghép detection khi `propagation.flow = dam4sam`.

Khác flow (dời box theo chuyển động điểm ảnh, cần detector để bám sát vật), DAM4SAM phân đoạn chính vật đó ở mỗi ảnh
(mask -> box) nên bám được cả khi detector sót hoặc vật bị che một phần; bù lại cần GPU (SAM 2.1 hiera-large ~1 s/ảnh
cho mỗi vật trên RTX 3090 ở 1600x900) nên chỉ hợp khi chạy nền / theo lô, không hợp cho bấm "Lan truyền" trên web máy
không có GPU.

Cài: xem đầu tools2d/dam4sam.py (clone repo DAM4SAM vào tools2d/DAM4SAM, pip install -e ., tải checkpoint). Khi chưa
cài, `Dam4SamPredictor` báo lỗi rõ lúc tạo chứ không lúc đang lan truyền.

Tracker (src/services/propagation.py) gọi `predict(track, image_path)` cho từng track còn sống ở mỗi ảnh; trả về box
hoặc None (mask rỗng = vật khuất / ra khỏi ảnh, tracker chuyển sang dự đoán theo vận tốc). Mỗi track là một
DAM4SAMTracker riêng, khởi tạo bằng box người chốt ở keyframe; nhiều track trên cùng ảnh thì ảnh chỉ đọc một lần.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

INSTALL_HINT = "cài theo hướng dẫn ở đầu tools2d/dam4sam.py (clone repo DAM4SAM vào tools2d/DAM4SAM, pip install -e ., tải checkpoint)"


def mask_to_box(mask: np.ndarray, min_px: int = 4) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_px:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _load_dam4sam(repo_dir: str | Path | None):
    """Import DAM4SAMTracker từ bản clone của repo DAM4SAM (không có trên PyPI)."""
    import os
    import sys

    repo = Path(repo_dir or os.environ.get("DAM4SAM_DIR") or Path("tools2d") / "DAM4SAM").resolve()
    if not (repo / "dam4sam_tracker.py").is_file():
        raise RuntimeError(f"Không thấy repo DAM4SAM ở {repo}. {INSTALL_HINT}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from dam4sam_tracker import DAM4SAMTracker
    except ImportError as e:
        raise RuntimeError(f"Import DAM4SAM lỗi ({e}). {INSTALL_HINT}") from e
    return DAM4SAMTracker


_TEMPLATES: dict[tuple[str, str], object] = {}


def _template(model: str, repo_dir: str | Path | None):
    """Tracker mẫu (giữ mô hình SAM 2.1 trên GPU), nhớ theo (model, repo): mỗi lần lan truyền / mỗi scene khi đánh giá
    tạo một Dam4SamPredictor mới, không được nạp lại ~900 MB weights mỗi lần."""
    key = (model, str(repo_dir or ""))
    if key not in _TEMPLATES:
        _TEMPLATES[key] = _load_dam4sam(repo_dir)(model)
    return _TEMPLATES[key]


def _clone_tracker(template):
    """Mỗi vật cần một DAM4SAMTracker (trạng thái bộ nhớ riêng) nhưng chỉ nạp mô hình SAM 2.1 một lần: sao chép
    tracker mẫu, dùng chung `predictor` (mô hình), trạng thái suy luận được tạo mới ở initialize()."""
    import copy

    tr = copy.copy(template)
    tr.tracking_times = []
    return tr


class Dam4SamPredictor:
    """Một DAM4SAM tracker cho mỗi track; dự đoán box ở ảnh mới từ mask SAM 2.1."""

    def __init__(self, resolve: Callable[[str], Path], model: str = "sam21pp-L", repo_dir: str | Path | None = None,
                 tracker_factory=None):  # fmt: skip
        """model: tên trong DAM4SAM ("sam21pp-L" = SAM 2.1 hiera-large, "sam21pp-B"/"-S"/"-T" nhỏ hơn, nhanh hơn).
        repo_dir: bản clone repo DAM4SAM (mặc định tools2d/DAM4SAM hoặc biến môi trường DAM4SAM_DIR), checkpoint nằm ở
        <repo>/checkpoints. tracker_factory(): trả về object có initialize(img, None, bbox=[x, y, w, h]) và
        track(img) -> {'pred_mask'} — test truyền bản giả, không cần GPU."""
        self.resolve = resolve
        self._img: dict[str, object] = {}
        self._trackers: dict[str, object] = {}
        self.calls = 0
        if tracker_factory is not None:
            self.factory = tracker_factory
            return
        template = _template(model, repo_dir)  # nạp SAM 2.1 lên GPU một lần cho cả tiến trình
        self.factory = lambda: _clone_tracker(template)

    def image(self, path: str):
        """Ảnh PIL RGB (giao diện của DAM4SAM); nhớ 3 ảnh gần nhất."""
        if path not in self._img:
            if len(self._img) > 2:
                self._img.pop(next(iter(self._img)))
            p = self.resolve(path)
            if not Path(p).is_file():
                self._img[path] = None
            else:
                from PIL import Image

                self._img[path] = Image.open(p).convert("RGB")
        return self._img[path]

    def start(self, track_id: str, image_path: str, box: list[float]) -> bool:
        img = self.image(image_path)
        if img is None:
            return False
        x1, y1, x2, y2 = (float(v) for v in box)
        tr = self.factory()
        tr.initialize(img, None, bbox=[x1, y1, x2 - x1, y2 - y1])
        self._trackers[track_id] = tr
        return True

    def predict(self, track_id: str, image_path: str) -> list[float] | None:
        tr = self._trackers.get(track_id)
        img = self.image(image_path) if tr is not None else None
        if tr is None or img is None:
            return None
        self.calls += 1
        try:
            out = tr.track(img)
        except Exception as e:  # pragma: no cover - lỗi GPU / model
            log.warning("DAM4SAM lỗi ở %s: %s", image_path, e)
            return None
        mask = out.get("pred_mask") if isinstance(out, dict) else out
        return mask_to_box(np.asarray(mask)) if mask is not None else None

    def stop(self, track_id: str) -> None:
        self._trackers.pop(track_id, None)
