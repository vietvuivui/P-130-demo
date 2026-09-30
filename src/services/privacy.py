"""Làm mờ mặt người và biển số trước khi ảnh rời máy chủ (FR-03).

Ảnh được làm mờ lần đầu khi có người xem (hoặc khi xuất), cache ở <workspace>/anon/<khoá>.jpg; khoá gồm đường dẫn,
thời điểm sửa file và cấu hình, nên đổi ảnh hay đổi ngưỡng thì tự làm lại. File gốc không bị sửa (detector và QA Agent
vẫn chạy trên ảnh gốc).

- Biển số: open-image-models (YOLOv9 ONNX chuyên cho biển số). Chạy trên cả ảnh và 4 ô cắt chồng nhau, vì biển số ở xa
  chỉ vài chục pixel.
- Mặt người: YOLOE với prompt "human face", cộng thêm vùng đầu của mỗi người đủ lớn (người thì detector bắt rất chắc,
  mặt nhỏ thì không): chấp nhận làm mờ thừa một chút để không sót.
- Thiếu thư viện / lỗi đọc ảnh: trả ảnh gốc và ghi cảnh báo một lần, không làm hỏng việc duyệt.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np

from src.models.qa_config import PrivacyCfg

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_MODELS: dict[str, object] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg)


def _plate_detector(cfg: PrivacyCfg):
    key = "plate:" + cfg.plate_model
    if key not in _MODELS:
        try:
            import open_image_models as oim

            if hasattr(oim, "create_detector"):
                _MODELS[key] = oim.create_detector(cfg.plate_model, conf_thresh=cfg.plate_conf)
            else:  # bản cũ
                _MODELS[key] = oim.LicensePlateDetector(detection_model=cfg.plate_model, conf_thresh=cfg.plate_conf)
        except Exception as e:  # chưa cài / không tải được model
            _warn_once(key, f"Không làm mờ được biển số (open-image-models: {e}); cài: pip install open-image-models")
            _MODELS[key] = None
    return _MODELS[key]


def _face_model(cfg: PrivacyCfg):
    key = "yoloe:" + ",".join(cfg.face_prompts)
    if key not in _MODELS:
        try:
            from ultralytics import YOLOE

            m = YOLOE("yoloe-26l-seg.pt")
            names = [*cfg.face_prompts, "person"]
            m.set_classes(names, m.get_text_pe(names))
            _MODELS[key] = (m, names)
        except Exception as e:
            _warn_once(key, f"Không làm mờ được mặt người (YOLOE: {e})")
            _MODELS[key] = None
    return _MODELS[key]


def _tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """4 ô cắt chồng nhau 20%, cộng cả ảnh."""
    tw, th = int(w * 0.6), int(h * 0.6)
    return [(0, 0, w, h)] + [(x, y, x + tw, y + th) for x in (0, w - tw) for y in (0, h - th)]


def detect_regions(img: np.ndarray, cfg: PrivacyCfg) -> list[tuple[float, float, float, float, str]]:
    """Vùng cần làm mờ (x1, y1, x2, y2, loại) trên ảnh BGR."""
    h, w = img.shape[:2]
    out = []
    with _LOCK:
        plate = _plate_detector(cfg)
        if plate is not None:
            for x0, y0, x1, y1 in _tiles(w, h) if cfg.plate_tiles else [(0, 0, w, h)]:
                for d in plate.predict(np.ascontiguousarray(img[y0:y1, x0:x1])):
                    b = d.bounding_box
                    out.append((b.x1 + x0, b.y1 + y0, b.x2 + x0, b.y2 + y0, "plate"))
        face = _face_model(cfg)
        if face is not None:
            model, names = face
            r = model.predict(img, conf=min(cfg.face_conf, cfg.person_conf), imgsz=cfg.imgsz, verbose=False)[0]
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                               strict=True):  # fmt: skip
                if names[int(c)] != "person":
                    if s >= cfg.face_conf:
                        out.append((x1, y1, x2, y2, "face"))
                elif s >= cfg.person_conf and y2 - y1 >= cfg.min_person_px:
                    bh, bw = y2 - y1, x2 - x1
                    head_w = min(bw, bh * 0.3)
                    cx = (x1 + x2) / 2
                    out.append((cx - head_w / 2, y1, cx + head_w / 2, y1 + bh * 0.18, "head"))
    return out


def blur_regions(img: np.ndarray, regions, pad: float) -> np.ndarray:
    """Pixel hoá rồi làm nhoè từng vùng (không khôi phục lại được bằng làm nét)."""
    import cv2

    out = img.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2, _ in regions:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
        c, d = min(w, int(x2 + px + 1)), min(h, int(y2 + py + 1))
        if c - a < 2 or d - b < 2:
            continue
        roi = out[b:d, a:c]
        small = cv2.resize(roi, (max(1, (c - a) // 8), max(1, (d - b) // 8)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)
        k = max(3, ((min(c - a, d - b) // 4) | 1))
        out[b:d, a:c] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def _key(src: Path, cfg: PrivacyCfg) -> str:
    st = src.stat()
    raw = f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{cfg.model_dump_json()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def anonymized_path(cache_root: Path, src: Path, cfg: PrivacyCfg, detector=None) -> Path:
    """Đường dẫn ảnh đã làm mờ (tạo nếu chưa có); ảnh gốc nếu tắt / không làm được."""
    if not cfg.enabled or not src.exists():
        return src
    try:
        dst = Path(cache_root) / "anon" / f"{_key(src, cfg)}.jpg"
        if dst.exists():
            return dst
        import cv2

        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return src
        regions = (detector or detect_regions)(img, cfg)
        out = blur_regions(img, regions, cfg.pad) if regions else img
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return src
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(dst)
        return dst
    except Exception as e:  # không để việc làm mờ làm hỏng việc xem ảnh
        _warn_once("anon-error", f"Làm mờ ảnh lỗi, gửi ảnh gốc: {e}")
        return src


def image_loader(store, cfg: PrivacyCfg | None = None):
    """Hàm path -> PIL.Image đã làm mờ, cho các bộ xuất có chép ảnh (KITTI)."""
    from PIL import Image

    if cfg is None:
        from src.config import get_settings
        from src.models.qa_config import get_autolabel_config

        cfg = get_autolabel_config(get_settings().autolabel_config).privacy

    def load(path: Path):
        return Image.open(anonymized_path(store.root, Path(path), cfg))

    return load
