"""Bấm một điểm lên vật -> mask + box (thay cho việc kéo vẽ box): `segment_points(image_path, points)`.

Ba cách, tự chọn theo máy chủ (ưu tiên từ trên xuống):
- **sam2-onnx**: SAM 2.1 xuất sang ONNX, chạy bằng ONNX Runtime (CUDA nếu có, không thì CPU) — không cần PyTorch hay
  repo DAM4SAM. Cách chạy theo AnyLabeling (github.com/vietanhdev/anylabeling): encoder và decoder tách riêng, embedding
  của ảnh được nhớ (LRU) nên chỉ lần bấm đầu trên một ảnh phải chờ encoder (Hiera-Tiny trên CPU: ~2 s cho ảnh 2560x1440,
  các lần bấm sau ~0.1–0.2 s). Mô hình: `python scripts/download_sam_onnx.py` (tải vào weights/sam2-onnx/).
  Nhận điểm dương / âm và box thô (kéo một hình chữ nhật quanh vật -> mask bó sát).
- **sam2**: SAM 2.1 (bản đi kèm repo DAM4SAM ở tools2d/DAM4SAM, dùng chung mô hình với luồng lan truyền DAM4SAM nên
  không nạp thêm weights). Điểm dương = thuộc vật, điểm âm = không thuộc; mỗi lần bấm thêm là tinh chỉnh lại. Ảnh được
  mã hoá một lần (~0.3–1 s trên GPU), các lần bấm sau trên cùng ảnh chỉ mất vài chục ms.
- **grabcut**: OpenCV GrabCut quanh điểm bấm, chạy CPU, không cần cài gì. Kém hơn SAM rõ (dựa vào màu), chỉ để máy
  không có GPU vẫn dùng được công cụ; kết quả ghi `engine: grabcut` để UI báo cho người dùng biết.

Trả về box [x1, y1, x2, y2] và đa giác viền mask (đã giản lược) theo pixel ảnh gốc.

Phần ONNX viết lại theo đầu vào / đầu ra của chính mô hình (image 1x3x1024x1024 chuẩn hoá ImageNet -> image_embed,
high_res_feats_0/1; decoder nhận point_coords theo hệ 1024, point_labels với 2 / 3 là hai góc box); không chép mã của
AnyLabeling (GPL-3.0).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_SAM: dict = {"predictor": None, "path": None, "failed": None}
MAX_POLY_POINTS = 60


class SegmentError(ValueError):
    pass


# ---------------------------------------------------------------------------
# SAM 2.1 ONNX
# ---------------------------------------------------------------------------

ONNX_DIR = Path(__file__).resolve().parents[2] / "weights" / "sam2-onnx"
ONNX_SIZE = 1024
EMB_CACHE = 8  # số ảnh giữ embedding (mỗi ảnh ~13 MB)
_ONNX: dict = {"enc": None, "dec": None, "name": None, "failed": None, "cache": OrderedDict(), "lock": threading.Lock()}


def onnx_model_files() -> tuple[Path, Path] | None:
    """(encoder, decoder) trong weights/sam2-onnx (hoặc SAM_ONNX_DIR); nhiều bản thì lấy bản lớn nhất."""
    d = Path(os.environ.get("SAM_ONNX_DIR") or ONNX_DIR)
    encs = sorted(d.glob("**/*.encoder.onnx"), key=lambda p: p.stat().st_size, reverse=True)
    for enc in encs:
        dec = enc.with_name(enc.name.replace(".encoder.onnx", ".decoder.onnx"))
        if dec.is_file():
            return enc, dec
    return None


def onnx_available() -> str | None:
    """None nếu dùng được SAM 2.1 ONNX; không thì lý do."""
    if _ONNX["failed"]:
        return _ONNX["failed"]
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        return "chưa cài onnxruntime (pip install onnxruntime, hoặc onnxruntime-gpu cho CUDA)"
    if onnx_model_files() is None:
        return "chưa có mô hình: python scripts/download_sam_onnx.py"
    return None


def _onnx_sessions():
    if _ONNX["enc"] is None:
        import onnxruntime as ort

        enc_path, dec_path = onnx_model_files()
        want = os.environ.get("SAM_ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",")
        providers = [p for p in want if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _ONNX["enc"] = ort.InferenceSession(str(enc_path), so, providers=providers)
        _ONNX["dec"] = ort.InferenceSession(str(dec_path), so, providers=providers)
        _ONNX["name"] = f"{enc_path.name.replace('.encoder.onnx', '')} · {_ONNX['enc'].get_providers()[0].replace('ExecutionProvider', '')}"
        log.info("SAM ONNX: %s", _ONNX["name"])
    return _ONNX["enc"], _ONNX["dec"]


def _onnx_embedding(image_path: Path) -> tuple[tuple, int, int]:
    """(embedding, rộng, cao) của ảnh; nhớ LRU theo đường dẫn + thời điểm sửa file."""
    import cv2

    key = (str(image_path), image_path.stat().st_mtime_ns)
    cache: OrderedDict = _ONNX["cache"]
    with _ONNX["lock"]:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        enc, _ = _onnx_sessions()
        img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SegmentError("Không đọc được ảnh")
        h, w = img.shape[:2]
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (ONNX_SIZE, ONNX_SIZE)).astype(np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        out = enc.run(None, {enc.get_inputs()[0].name: x})
        named = {o.name: v for o, v in zip(enc.get_outputs(), out, strict=True)}
        cache[key] = (named, w, h)
        while len(cache) > EMB_CACHE:
            cache.popitem(last=False)
        return cache[key]


def preload(image_path: str | Path) -> bool:
    """Mã hoá ảnh trước ở luồng nền (UI gọi khi mở frame / bật công cụ) để lần bấm đầu không phải chờ encoder."""
    image_path = Path(image_path)
    if onnx_available() is not None or not image_path.is_file():
        return False

    def run():
        try:
            _onnx_embedding(image_path)
        except Exception:  # không để lỗi nền làm hỏng gì; lần bấm thật sẽ báo lỗi
            log.exception("SAM ONNX preload lỗi")

    threading.Thread(target=run, daemon=True, name="sam-preload").start()
    return True


def pick_mask(masks: np.ndarray, ious: np.ndarray, single_point: bool) -> int:
    """Chọn 1 trong các mask SAM trả về. Một điểm bấm thì mơ hồ (cả xe / kính xe / bánh xe): trong các mask đủ tin
    (IoU dự đoán >= max(0.7, cao nhất - 0.25)) lấy mask to nhất, vì ở đây người dùng gán nhãn cả vật. Nhiều điểm hoặc
    có box thì prompt đã rõ: lấy mask điểm cao nhất."""
    ious = np.asarray(ious).reshape(-1)
    if not single_point:
        return int(np.argmax(ious))
    cand = [i for i in range(len(ious)) if ious[i] >= max(0.7, float(ious.max()) - 0.25)]
    if not cand:
        return int(np.argmax(ious))
    return max(cand, key=lambda i: int((masks[i] > 0).sum()))


def _segment_onnx(image_path: Path, points: list[list[float]], labels: list[int], box: list[float] | None) -> dict:
    import cv2

    emb, w, h = _onnx_embedding(image_path)
    _, dec = _onnx_sessions()
    pts = [list(p) for p in points]
    lbs = [float(v) for v in labels]
    if box is not None:  # box thô = hai điểm góc với nhãn 2 (trên-trái) và 3 (dưới-phải)
        pts += [[box[0], box[1]], [box[2], box[3]]]
        lbs += [2.0, 3.0]
    pc = np.array(pts, np.float32)[None]
    pc[..., 0] *= ONNX_SIZE / w
    pc[..., 1] *= ONNX_SIZE / h
    feed = {
        **emb,
        "point_coords": pc,
        "point_labels": np.array(lbs, np.float32)[None],
        "mask_input": np.zeros((1, 1, ONNX_SIZE // 4, ONNX_SIZE // 4), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
    }
    names = {i.name for i in dec.get_inputs()}
    masks, ious = dec.run(None, {k: v for k, v in feed.items() if k in names})[:2]
    i = pick_mask(masks[0], ious[0], single_point=box is None and len(points) == 1)
    mask = cv2.resize(masks[0, i], (w, h), interpolation=cv2.INTER_LINEAR) > 0
    r = mask_to_result(mask, float(np.asarray(ious).reshape(-1)[i]), "sam2-onnx")
    r["model"] = _ONNX["name"]
    return r


def sam_available() -> str | None:
    """None nếu dùng được SAM 2.1; không thì lý do."""
    if _SAM["failed"]:
        return _SAM["failed"]
    from src.services import dam4sam

    problems = dam4sam.check_install()
    return "; ".join(problems) if problems else None


def engine_info() -> dict:
    onnx_reason = onnx_available()
    if onnx_reason is None:
        return {"engine": "sam2-onnx", "model": _ONNX["name"], "sam_unavailable": None}
    reason = sam_available()
    return {"engine": "grabcut" if reason else "sam2", "sam_unavailable": reason, "onnx_unavailable": onnx_reason}


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
                   engine: str | None = None, box: list[float] | None = None) -> dict:  # fmt: skip
    """points: [[x, y], ...] pixel ảnh gốc; labels: 1 = thuộc vật, 0 = không thuộc (mặc định toàn 1).
    box: [x1, y1, x2, y2] box thô quanh vật (tuỳ chọn; cần ít nhất một điểm hoặc box).
    engine: "sam2-onnx" | "sam2" | "grabcut" | None (tự chọn theo engine_info)."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise SegmentError("Không tìm thấy file ảnh")
    if not points and box is None:
        raise SegmentError("Cần ít nhất một điểm hoặc một box")
    labels = labels or [1] * len(points)
    if len(labels) != len(points):
        raise SegmentError("Số nhãn điểm không khớp số điểm")
    use = engine or engine_info()["engine"]
    if use == "sam2-onnx":
        try:
            return _segment_onnx(image_path, points, labels, box)
        except SegmentError:
            raise
        except Exception as e:
            log.exception("SAM ONNX lỗi, chuyển sang cách khác")
            _ONNX["failed"] = f"{type(e).__name__}: {e}"
            if engine == "sam2-onnx":
                raise SegmentError(f"SAM ONNX lỗi: {e}") from e
            use = engine_info()["engine"]
    if box is not None and not points:  # các cách còn lại không nhận box: dùng tâm box làm điểm
        points, labels = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], [1]
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
