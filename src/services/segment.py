"""Bấm một điểm lên vật -> mask + box (thay cho việc kéo vẽ box): `segment_points(image_path, points)`.

Hai cách, tự chọn theo máy chủ:
- **sam2**: SAM 2.1 (bản đi kèm repo DAM4SAM ở tools2d/DAM4SAM, dùng chung mô hình với luồng lan truyền DAM4SAM nên
  không nạp thêm weights). Điểm dương = thuộc vật, điểm âm = không thuộc; mỗi lần bấm thêm là tinh chỉnh lại. Ảnh được
  mã hoá một lần (~0.3–1 s trên GPU), các lần bấm sau trên cùng ảnh chỉ mất vài chục ms.
- **grabcut**: OpenCV GrabCut quanh điểm bấm, chạy CPU, không cần cài gì. Kém hơn SAM rõ (dựa vào màu), chỉ để máy
  không có GPU vẫn dùng được công cụ; kết quả ghi `engine: grabcut` để UI báo cho người dùng biết.

Trả về box [x1, y1, x2, y2] và đa giác viền mask (đã giản lược) theo pixel ảnh gốc.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_SAM: dict = {"predictor": None, "path": None, "failed": None}
MAX_POLY_POINTS = 60


class SegmentError(ValueError):
    pass


def sam_available() -> str | None:
    """None nếu dùng được SAM 2.1; không thì lý do."""
    if _SAM["failed"]:
        return _SAM["failed"]
    from src.services import dam4sam

    problems = dam4sam.check_install()
    return "; ".join(problems) if problems else None


def engine_info() -> dict:
    reason = sam_available()
    return {"engine": "grabcut" if reason else "sam2", "sam_unavailable": reason}


def mask_to_result(mask: np.ndarray, score: float, engine: str) -> dict:
    """Mask nhị phân -> {bbox, polygon, score, engine}; chỉ lấy vùng liên thông lớn nhất."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SegmentError("Không tách được vật ở điểm này, thử bấm vào giữa vật hoặc vẽ box")
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 3 or h < 3:
        raise SegmentError("Vùng tách được quá nhỏ, thử bấm vào giữa vật hoặc vẽ box")
    eps = 0.004 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    while len(poly) > MAX_POLY_POINTS:
        eps *= 1.5
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return {
        "bbox": [float(x), float(y), float(x + w), float(y + h)],
        "polygon": [float(v) for v in poly.reshape(-1)] if len(poly) >= 3 else None,
        "score": round(float(score), 3),
        "engine": engine,
        "area": int(m.sum()),
    }


def _sam_predictor():
    if _SAM["predictor"] is None:
        from src.services import dam4sam

        template = dam4sam._template("sam21pp-L", None)  # dùng chung mô hình với luồng lan truyền DAM4SAM
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _SAM["predictor"] = SAM2ImagePredictor(template.predictor)
    return _SAM["predictor"]


def _segment_sam(image_path: Path, points: list[list[float]], labels: list[int]) -> dict:
    import torch
    from PIL import Image

    with _LOCK:  # một ảnh được mã hoá tại một thời điểm; các lần bấm tiếp trên cùng ảnh dùng lại embedding
        pred = _sam_predictor()
        if _SAM["path"] != str(image_path):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred.set_image(np.array(Image.open(image_path).convert("RGB")))
            _SAM["path"] = str(image_path)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = pred.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=len(points) == 1,  # một điểm thì mơ hồ (cả xe / bánh xe): lấy mask điểm cao nhất
            )
    i = int(np.argmax(scores))
    return mask_to_result(masks[i], float(scores[i]), "sam2")


def _segment_grabcut(image_path: Path, points: list[list[float]], labels: list[int]) -> dict:
    import cv2

    img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SegmentError("Không đọc được ảnh")
    h, w = img.shape[:2]
    pos = [p for p, lb in zip(points, labels, strict=True) if lb == 1]
    if not pos:
        raise SegmentError("Cần ít nhất một điểm thuộc vật")
    # Thu ảnh cho nhanh; vùng xét = hộp quanh các điểm dương, rộng ~1/3 ảnh mỗi phía
    s = min(1.0, 640 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    xs, ys = [p[0] * s for p in pos], [p[1] * s for p in pos]
    rx, ry = sw / 3, sh / 3
    x1, y1 = int(max(0, min(xs) - rx)), int(max(0, min(ys) - ry))
    x2, y2 = int(min(sw - 1, max(xs) + rx)), int(min(sh - 1, max(ys) + ry))
    mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    r = max(3, int(min(sw, sh) * 0.012))
    for p, lb in zip(points, labels, strict=True):
        c = (int(p[0] * s), int(p[1] * s))
        cv2.circle(mask, c, r * 3 if lb == 1 else r * 2, cv2.GC_PR_FGD if lb == 1 else cv2.GC_BGD, -1)
        if lb == 1:
            cv2.circle(mask, c, r, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    # Làm mờ nhẹ trước: nhiễu JPEG ở mép vật làm GrabCut cắt hụt vài pixel
    cv2.grabCut(cv2.GaussianBlur(small, (5, 5), 0), mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Chỉ giữ vùng liên thông chứa điểm dương đầu tiên
    n, lab = cv2.connectedComponents(fg)
    k = lab[min(sh - 1, int(ys[0])), min(sw - 1, int(xs[0]))]
    fg = (lab == k).astype(np.uint8) if n > 1 and k > 0 else fg
    if s < 1:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_to_result(fg, 0.5, "grabcut")


def segment_points(image_path: str | Path, points: list[list[float]], labels: list[int] | None = None,
                   engine: str | None = None) -> dict:  # fmt: skip
    """points: [[x, y], ...] pixel ảnh gốc; labels: 1 = thuộc vật, 0 = không thuộc (mặc định toàn 1).
    engine: "sam2" | "grabcut" | None (tự chọn: SAM nếu máy chủ có, không thì GrabCut)."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise SegmentError("Không tìm thấy file ảnh")
    if not points:
        raise SegmentError("Cần ít nhất một điểm")
    labels = labels or [1] * len(points)
    if len(labels) != len(points):
        raise SegmentError("Số nhãn điểm không khớp số điểm")
    use = engine or engine_info()["engine"]
    if use == "sam2":
        try:
            return _segment_sam(image_path, points, labels)
        except SegmentError:
            raise
        except Exception as e:  # thiếu GPU / lỗi nạp mô hình: ghi lại, lần sau dùng GrabCut luôn
            log.exception("SAM 2.1 lỗi, chuyển sang GrabCut")
            _SAM["failed"] = f"{type(e).__name__}: {e}"
            if engine == "sam2":
                raise SegmentError(f"SAM 2.1 lỗi: {e}") from e
    return _segment_grabcut(image_path, points, labels)
