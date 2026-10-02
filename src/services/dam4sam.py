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


CHECKPOINTS = {"sam21pp-L": "sam2.1_hiera_large.pt", "sam21pp-B": "sam2.1_hiera_base_plus.pt"}
_ROOT = Path(__file__).resolve().parents[2]


def check_install(model: str = "sam21pp-L") -> list[str]:
    """Những thứ còn thiếu để chạy DAM4SAM (rỗng = đủ). Không nạp mô hình."""
    import importlib.util
    import os

    repo = Path(os.environ.get("DAM4SAM_DIR") or _ROOT / "tools2d" / "DAM4SAM")
    if not (repo / "dam4sam_tracker.py").is_file():
        return [f"repo DAM4SAM ở {repo}: git clone https://github.com/jovanavidenovic/DAM4SAM.git tools2d/DAM4SAM"]
    problems = []
    needs_vot = "from vot" in (repo / "dam4sam_tracker.py").read_text(encoding="utf-8", errors="ignore")
    mods = [("sam2", f"cd {repo} rồi pip install -e ."), ("torch", "pip install torch (bản CUDA)"), ("yaml", "pip install pyyaml")]
    if needs_vot:  # bản clone chưa thay `vot` bằng vot_shim.py
        mods.append(("vot", "pip install vot-toolkit==0.7.1 vot-trax==4.0.2"))
    for mod, hint in mods:
        if importlib.util.find_spec(mod) is None:
            problems.append(f"gói `{mod}`: {hint}")
    ckpt = CHECKPOINTS.get(model)
    if ckpt and not (repo / "checkpoints" / ckpt).is_file():
        problems.append(
            f"checkpoint {repo / 'checkpoints' / ckpt}: curl.exe -L -o {repo / 'checkpoints' / ckpt} "
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt}"
        )
    if importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            problems.append("CUDA: torch không thấy GPU (DAM4SAM nạp mô hình lên cuda:0)")
    return problems


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


class SharedEncoder:
    """Bọc `predictor.forward_image` (image encoder Hiera — phần nặng nhất của SAM 2.1): khi `key` được đặt (đường dẫn
    ảnh đang xử lý) thì ảnh đó chỉ được mã hoá một lần, các vật khác trên cùng ảnh dùng lại kết quả. Bản gốc mã hoá lại
    cho từng vật vì mỗi vật là một DAM4SAMTracker riêng. `key = None`: chạy như gốc."""

    def __init__(self, orig):
        self.orig = orig
        self.key = None
        self._cached_key = None
        self._out = None
        self.hits = 0
        self.misses = 0

    def __call__(self, img_batch):
        if self.key is not None and self.key == self._cached_key:
            self.hits += 1
            return self._out
        out = self.orig(img_batch)
        self.misses += 1
        self._cached_key, self._out = self.key, out
        return out


class _Fast:
    """Ngữ cảnh cho một lần gọi DAM4SAM: đặt khoá ảnh cho SharedEncoder, bật autocast, bỏ torch.cuda.empty_cache()
    (DAM4SAM gọi nó ở mỗi ảnh của mỗi vật, chậm mà không cần khi các tracker dùng chung mô hình)."""

    def __init__(self, encoder: SharedEncoder | None, key: str | None, autocast: bool):
        self.encoder, self.key, self.autocast = encoder, key, autocast
        self._ctx = None
        self._empty = None

    def __enter__(self):
        if self.encoder is not None:
            self.encoder.key = self.key
        try:
            import torch
        except ImportError:  # test với tracker giả
            return self
        self._empty = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        if self.autocast and torch.cuda.is_available():
            self._ctx = torch.autocast("cuda")  # như @torch.cuda.amp.autocast() trong script gốc của DAM4SAM
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        if self._empty is not None:
            import torch

            torch.cuda.empty_cache = self._empty
        if self.encoder is not None:
            self.encoder.key = None
        return False


_ENCODERS: dict[int, SharedEncoder] = {}
_AUTOCAST_OK: dict[int, bool] = {}


def _shared_encoder(template) -> SharedEncoder | None:
    """Gắn SharedEncoder vào mô hình của tracker mẫu (một lần)."""
    pred = getattr(template, "predictor", None)
    if pred is None or not hasattr(pred, "forward_image"):
        return None
    if id(pred) not in _ENCODERS:
        enc = SharedEncoder(pred.forward_image)
        pred.forward_image = enc
        _ENCODERS[id(pred)] = enc
    return _ENCODERS[id(pred)]


class Dam4SamPredictor:
    """Một DAM4SAM tracker cho mỗi track; dự đoán box ở ảnh mới từ mask SAM 2.1."""

    def __init__(self, resolve: Callable[[str], Path], model: str = "sam21pp-L", repo_dir: str | Path | None = None,
                 tracker_factory=None, share_encoder: bool = True, autocast: bool = True):  # fmt: skip
        """model: tên trong DAM4SAM ("sam21pp-L" = SAM 2.1 hiera-large, "sam21pp-B"/"-S"/"-T" nhỏ hơn, nhanh hơn).
        repo_dir: bản clone repo DAM4SAM (mặc định tools2d/DAM4SAM hoặc biến môi trường DAM4SAM_DIR), checkpoint nằm ở
        <repo>/checkpoints. tracker_factory(): trả về object có initialize(img, None, bbox=[x, y, w, h]) và
        track(img) -> {'pred_mask'} — test truyền bản giả, không cần GPU.
        share_encoder / autocast: tăng tốc (xem SharedEncoder, _Fast); tắt cả hai = chạy đúng như repo gốc."""
        self.resolve = resolve
        self._img: dict[str, object] = {}
        self._trackers: dict[str, object] = {}
        self.calls = 0
        self.encoder: SharedEncoder | None = None
        self.autocast = autocast
        self._template = None
        if tracker_factory is not None:
            self.factory = tracker_factory
            self.autocast = False
            return
        template = _template(model, repo_dir)  # nạp SAM 2.1 lên GPU một lần cho cả tiến trình
        self._template = template
        enc = _shared_encoder(template)
        self.encoder = enc if share_encoder else None
        self.factory = lambda: _clone_tracker(template)

    def _fast(self, image_path: str) -> _Fast:
        return _Fast(self.encoder, image_path, self.autocast)

    def _probe_autocast(self, img) -> None:
        """Thử autocast một lần trên tracker tạm (khởi tạo + theo dõi chính ảnh đó). Lỗi (GPU / bản torch không hợp)
        thì tắt autocast cho cả tiến trình thay vì hỏng giữa chừng một lần lan truyền."""
        key = id(getattr(self._template, "predictor", self._template))
        if not self.autocast or self._template is None or key in _AUTOCAST_OK:
            self.autocast = self.autocast and _AUTOCAST_OK.get(key, True)
            return
        try:
            tr = self.factory()
            w, h = img.size
            with _Fast(None, None, True):
                tr.initialize(img, None, bbox=[w * 0.4, h * 0.4, w * 0.2, h * 0.2])
                tr.track(img)
            _AUTOCAST_OK[key] = True
        except Exception as e:  # pragma: no cover - phụ thuộc GPU
            log.warning("DAM4SAM: autocast không dùng được (%s), chạy float32", e)
            _AUTOCAST_OK[key] = False
            self.autocast = False

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
        self._probe_autocast(img)
        x1, y1, x2, y2 = (float(v) for v in box)
        tr = self.factory()
        with self._fast(image_path):
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
            with self._fast(image_path):
                out = tr.track(img)
        except Exception as e:  # pragma: no cover - lỗi GPU / model
            log.warning("DAM4SAM lỗi ở %s: %s", image_path, e)
            return None
        mask = out.get("pred_mask") if isinstance(out, dict) else out
        return mask_to_box(np.asarray(mask)) if mask is not None else None

    def stop(self, track_id: str) -> None:
        self._trackers.pop(track_id, None)
