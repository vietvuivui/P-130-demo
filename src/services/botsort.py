"""BoT-SORT (Aharon et al. 2022) cho tracker lan truyền 2D: hai phần thêm vào ByteTrack.

1. Bù chuyển động camera (GMC): ước lượng phép biến đổi affine toàn ảnh giữa hai ảnh liền nhau (ORB + RANSAC) và
   dời box dự đoán theo đó, để xe lắc / rẽ không làm box dự đoán lệch khỏi vật đứng yên. Chỉ cần khi không dùng
   optical flow (flow đã dời từng box theo chuyển động thật, mạnh hơn GMC).
2. Ngoại hình (ReID): mỗi track giữ một vector đặc trưng cập nhật trượt (EMA, alpha 0.9); chi phí ghép =
   min(1 - IoU, khoảng cách ngoại hình), hai vế bị chặn (= 1) khi vượt ngưỡng (`fuse_cost`). Bài gốc dùng mạng ReID
   huấn luyện riêng; ở đây không có mạng ReID cho xe / người trên nuScenes nên dùng đặc trưng màu nhẹ (histogram HSV
   + ảnh thu nhỏ) chạy CPU, không cần huấn luyện. Thay bằng mạng ReID chỉ cần đổi `Appearance.embed`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Đếm để biết ngoại hình có thật sự đổi kết quả ghép không (tools2d/dam4sam.py in ra): calls = số lần ghép,
# changed = số lần kết quả khác ghép theo IoU thuần, rescued = cặp nhận nhờ ngoại hình dù IoU < ngưỡng
STATS = {"calls": 0, "changed": 0, "rescued": 0}
EMB_DIM = 8 * 8 + 8 + 6 * 6  # histogram H x S (8x8), histogram V (8 ô), ảnh xám thu nhỏ 6x6


def _cv2():
    import cv2

    return cv2


def load_bgr(path: str | Path, scale: float) -> np.ndarray | None:
    if not Path(path).is_file():
        return None
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# GMC
# ---------------------------------------------------------------------------


@dataclass
class Affine:
    """Phép biến đổi 2x3 (toạ độ pixel ảnh gốc) từ ảnh trước sang ảnh sau."""

    m: np.ndarray

    def warp_box(self, box: Sequence[float]) -> list[float]:
        x1, y1, x2, y2 = (float(v) for v in box)
        pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        q = pts @ self.m[:, :2].T + self.m[:, 2]
        return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def estimate_affine(gray_a: np.ndarray, gray_b: np.ndarray, scale: float, max_features: int = 1000) -> Affine | None:
    """Affine toàn ảnh a -> b bằng ORB + RANSAC như BoT-SORT (bản ORB; bài gốc còn có ECC, chậm hơn nhiều)."""
    cv2 = _cv2()
    orb = cv2.ORB_create(max_features)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None or inliers is None or inliers.sum() < 8:
        return None
    # Đổi về toạ độ ảnh gốc: M' = S^-1 M S với S = scale
    m = m.copy()
    m[:, 2] /= scale
    return Affine(m)


class GlobalMotion:
    """GMC giữa hai ảnh theo đường dẫn; nhớ vài ảnh xám gần nhất."""

    def __init__(self, resolve: Callable[[str], Path], scale: float = 0.5):
        self.resolve = resolve
        self.scale = scale
        self._gray: dict[str, np.ndarray | None] = {}

    def gray(self, path: str) -> np.ndarray | None:
        if path not in self._gray:
            if len(self._gray) > 8:
                self._gray.pop(next(iter(self._gray)))
            from src.services.flow import load_gray

            self._gray[path] = load_gray(self.resolve(path), self.scale)
        return self._gray[path]

    def between(self, path_a: str, path_b: str) -> Affine | None:
        a, b = self.gray(path_a), self.gray(path_b)
        if a is None or b is None or a.shape != b.shape:
            return None
        return estimate_affine(a, b, self.scale)


# ---------------------------------------------------------------------------
# Ngoại hình
# ---------------------------------------------------------------------------


def embed_crop(bgr: np.ndarray) -> np.ndarray | None:
    """Vector đặc trưng (chuẩn hoá L2) của một vùng ảnh: histogram H x S, histogram V, ảnh xám 6x6."""
    if bgr is None or bgr.size == 0 or bgr.shape[0] < 2 or bgr.shape[1] < 2:
        return None
    cv2 = _cv2()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    thumb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (6, 6), interpolation=cv2.INTER_AREA).ravel()
    thumb = thumb.astype(np.float64) - thumb.mean()
    # Căn bậc hai histogram (khoảng cách Hellinger) để vài ô trội không lấn át; độ sáng và hình dạng trọng số 0.5
    parts = [np.sqrt(hs / max(hs.sum(), 1e-9)), np.sqrt(v / max(v.sum(), 1e-9)) * 0.5,
             thumb / max(np.linalg.norm(thumb), 1e-9) * 0.5]  # fmt: skip
    f = np.concatenate(parts).astype(np.float64)
    return f / max(np.linalg.norm(f), 1e-9)


class Appearance:
    """Đặc trưng ngoại hình của các box trên một ảnh (đọc ảnh ở độ phân giải `scale`)."""

    def __init__(self, resolve: Callable[[str], Path], scale: float = 0.5):
        self.resolve = resolve
        self.scale = scale
        self._img: dict[str, np.ndarray | None] = {}

    def image(self, path: str) -> np.ndarray | None:
        if path not in self._img:
            if len(self._img) > 4:
                self._img.pop(next(iter(self._img)))
            self._img[path] = load_bgr(self.resolve(path), self.scale)
        return self._img[path]

    def features(self, path: str | None, boxes: Sequence[Sequence[float]]) -> list[np.ndarray | None]:
        img = self.image(path) if path else None
        if img is None:
            return [None] * len(boxes)
        h, w = img.shape[:2]
        out = []
        for b in boxes:
            x1, y1, x2, y2 = (float(v) * self.scale for v in b)
            # Lõi của box (bỏ 15% mép) để nền ít lẫn vào đặc trưng
            bw, bh = x2 - x1, y2 - y1
            x1, x2 = x1 + 0.15 * bw, x2 - 0.15 * bw
            y1, y2 = y1 + 0.15 * bh, y2 - 0.15 * bh
            c1, c2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))
            r1, r2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))
            out.append(embed_crop(img[r1:r2, c1:c2]) if c2 > c1 and r2 > r1 else None)
        return out


def cosine_distance(track_feats: Sequence[np.ndarray | None], det_feats: Sequence[np.ndarray | None]) -> np.ndarray:
    """Ma trận khoảng cách cosine (0..2); cặp thiếu đặc trưng = 1 (không giúp cũng không hại)."""
    m = np.ones((len(track_feats), len(det_feats)))
    for i, a in enumerate(track_feats):
        if a is None:
            continue
        for j, b in enumerate(det_feats):
            if b is not None:
                m[i, j] = 1.0 - float(a @ b)
    return m


def fuse_cost(iou: np.ndarray, app_dist: np.ndarray, proximity_thresh: float, appearance_thresh: float) -> np.ndarray:
    """Chi phí ghép của BoT-SORT: min(1 - IoU, khoảng cách ngoại hình); ngoại hình bị bỏ (= 1) khi quá khác hoặc khi
    hai box ở quá xa nhau (1 - IoU > proximity_thresh)."""
    iou_dist = 1.0 - iou
    emb = app_dist.copy()
    emb[emb > appearance_thresh] = 1.0
    emb[iou_dist > proximity_thresh] = 1.0
    return np.minimum(iou_dist, emb)


def update_feature(old: np.ndarray | None, new: np.ndarray | None, alpha: float) -> np.ndarray | None:
    """EMA đặc trưng của track như BoT-SORT: f = alpha f + (1 - alpha) f_det, chuẩn hoá lại."""
    if new is None:
        return old
    if old is None:
        return new
    f = alpha * old + (1 - alpha) * new
    return f / max(np.linalg.norm(f), 1e-9)
