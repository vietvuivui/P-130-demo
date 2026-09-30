"""Lan truyền nhãn 2D trên video (FR-11 → FR-13) — phần thuật toán, không phụ thuộc nuScenes.

Ý tưởng: người duyệt xong một keyframe; mọi quyết định của họ (giữ, sửa box, đổi lớp, vẽ thêm,
xoá) được mang sang các keyframe sau. Keyframe nuScenes chỉ có 2Hz nên box dịch nhiều giữa hai
keyframe; vì vậy tracker đi qua **mọi ảnh camera 12Hz** nằm giữa, ghép box dự đoán (vận tốc
không đổi trong toạ độ ảnh) với detection đã cache của từng ảnh, và chỉ ghi kết quả ở keyframe.

    keyframe K (đã approve) ── sweep ── sweep ── … ── keyframe K+1 ── sweep ── … ── keyframe K+2
          init_tracks            Tracker.step() ở mọi ảnh           apply_propagation() ở keyframe

Nguyên tắc giữ đúng human-in-the-loop:
- Lớp do người chốt được khoá theo track; detector ở frame sau chỉ góp hình học (box sát vật thể).
- Nhãn lan truyền vẫn ở trạng thái pending, người vẫn duyệt; c_prop cao thì rơi vào nhóm low
  (duyệt theo lô), c_prop thấp hoặc detector không thấy thì bị gắn issue để người xem kỹ.
- Object người đã xoá ở keyframe chỉ bị tự xoá ở frame sau khi khớp chặt và cùng lớp, được đánh dấu
  PROPAGATED_DELETE và bấm Keep là khôi phục được.
- Không bao giờ ghi vào frame người đã mở (status khác "auto"): lan truyền dừng trước frame đó.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.agents.nodes.issues import check_geometry
from src.agents.nodes.lidar import check_lidar
from src.agents.nodes.risk import score_risk
from src.models.qa_config import AutoLabelConfig, PropagationCfg
from src.models.schemas import (
    Detection,
    FrameRecord,
    LabelObject,
    PropagationInfo,
    QAIssue,
    QAResult,
    ReviewState,
)
from src.services.geometry import box_area, clip_box, iou, iou_matrix
from src.services.nuscenes_data import TimelineImage

# Box lan truyền được coi là "cùng object" với một box pre-label của frame đích khi IoU >= ngưỡng này
CLAIM_IOU = 0.5
# Mỗi ảnh không khớp detection, vận tốc bị giảm dần để box dự đoán không bay đi quá xa
COAST_VELOCITY_DAMPING = 0.8
PROPAGATION_REVIEWER = "propagation"
# Issue của nhóm "detection" nói về độ chắc của detector về lớp; với nhãn lan truyền, lớp đã được người
# chốt và độ chắc được thay bằng c_prop, nên các issue này không còn ý nghĩa
_DETECTOR_ISSUES = {"LOW_CONFIDENCE", "CLASS_CONFLICT"}


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: str
    label: str
    box: np.ndarray  # [x1, y1, x2, y2] ước lượng hiện tại
    kind: Literal["keep", "suppress"]
    keyframe_id: str
    keyframe_object_id: str
    last_t: int  # timestamp (µs) của ảnh cuối đã xử lý
    ref_points: int | None = None  # số điểm LiDAR trong box ở keyframe gốc
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))  # px/s cho từng cạnh
    misses: int = 0  # số ảnh liên tiếp không khớp detection
    steps: int = 0  # số ảnh đã đi qua
    data_steps: int = 0  # số ảnh có detection trong cache
    matched_steps: int = 0
    hops: int = 0  # số keyframe đã đi qua
    alive: bool = True
    stop_reason: str | None = None
    last_match: Detection | None = None  # detection khớp ở ảnh cuối (None = đang dự đoán)
    last_agreement: float = 0.0  # IoU giữa box dự đoán và detection ở ảnh cuối
    # Lớp detector gán ở lần khớp đầu tiên. Người có thể đã sửa lớp ở keyframe (detector nói car, người
    # chốt truck); chỉ khi lớp detector đổi giữa chừng mới là dấu hiệu track nhảy sang object khác
    first_det_label: str | None = None

    def stop(self, reason: str) -> None:
        self.alive = False
        self.stop_reason = reason

    @property
    def continuity(self) -> float | None:
        return self.matched_steps / self.data_steps if self.data_steps else None

    def output_box(self) -> list[float]:
        """Box ghi ra frame: detection nếu detector thấy (sát vật thể), không thì box dự đoán."""
        b = self.last_match.bbox if self.last_match is not None else self.box.tolist()
        return [round(float(v), 1) for v in b]


def count_points(uv: np.ndarray | None, box: list[float]) -> int | None:
    if uv is None or len(uv) == 0:
        return None
    x1, y1, x2, y2 = box
    return int(((uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)).sum())


def lidar_arrays(aux: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """LiDAR đã chiếu mà pipeline lưu cho UI ({u, v, d}) -> (uv (N, 2), depth (N,))."""
    if not aux or not aux.get("u"):
        return None, None
    uv = np.column_stack([np.asarray(aux["u"], dtype=np.float64), np.asarray(aux["v"], dtype=np.float64)])
    return uv, np.asarray(aux["d"], dtype=np.float64)


def init_tracks(keyframe: FrameRecord, lidar_aux: dict | None = None) -> list[Track]:
    """Tạo track từ quyết định của người ở keyframe. Gán track_id cho object của keyframe (sửa tại chỗ).

    - Object đã duyệt (Keep / sửa box / đổi lớp / vẽ thêm / duyệt theo lô) -> track "keep", lớp khoá theo người.
    - Object máy sinh bị người xoá -> track "suppress": detection cùng object ở frame sau sẽ được tự xoá.
    - Object còn pending không được lan truyền: chưa ai xác nhận.
    """
    uv, _ = lidar_arrays(lidar_aux)
    tracks = []
    for obj in keyframe.objects:
        r = obj.review
        if r.status == "approved":
            kind, box, label = "keep", r.final_bbox or obj.bbox, r.final_label or obj.label
        elif r.status == "deleted" and obj.source != "human":
            kind, box, label = "suppress", obj.bbox, obj.label
        else:
            continue
        obj.track_id = obj.track_id or f"{keyframe.frame_id}:{obj.object_id}"
        ref = count_points(uv, box)
        tracks.append(
            Track(
                track_id=obj.track_id,
                label=label,
                box=np.asarray(box, dtype=np.float64),
                kind=kind,
                keyframe_id=keyframe.frame_id,
                keyframe_object_id=obj.object_id,
                last_t=keyframe.image.timestamp,
                ref_points=ref or None,
            )
        )
    return tracks


def _greedy_match(pred: np.ndarray, dets: list[list[float]], thr: float) -> dict[int, tuple[int, float]]:
    """Ghép một-một theo IoU giảm dần. {chỉ số track: (chỉ số detection, IoU)}."""
    m = iou_matrix(pred, np.array(dets))
    rows, cols = np.where(m >= thr)
    out: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for i, j in sorted(zip(rows.tolist(), cols.tolist(), strict=True), key=lambda ij: -m[ij]):
        if i not in out and j not in used:
            out[i] = (j, float(m[i, j]))
            used.add(j)
    return out


class Tracker:
    """Theo dõi các track qua từng ảnh camera (keyframe hoặc sweep)."""

    def __init__(self, tracks: list[Track], cfg: PropagationCfg, width: int, height: int, motion=None,
                 start_path: str | None = None):  # fmt: skip
        """motion: FlowProvider (src/services/flow.py) để dự đoán box theo optical flow thay vì vận tốc không đổi,
        theo cfg.flow ("missing": chỉ ở ảnh chưa có detection; "always": mọi ảnh). start_path: ảnh của keyframe gốc."""
        self.tracks = tracks
        self.cfg = cfg
        self.width = width
        self.height = height
        self.images_without_detections = 0
        self.images_with_flow = 0
        self.motion = motion if cfg.flow != "off" else None
        self.prev_path = start_path

    @property
    def alive(self) -> list[Track]:
        return [t for t in self.tracks if t.alive]

    def step(self, image: TimelineImage, detections: list[Detection] | None) -> None:
        """Cập nhật mọi track còn sống bằng một ảnh. detections=None: ảnh chưa có trong cache."""
        tracks = self.alive
        if not tracks:
            return
        cfg = self.cfg
        field = None
        if self.motion is not None and self.prev_path and image.path and (cfg.flow == "always" or detections is None):
            field = self.motion.between(self.prev_path, image.path)
        self.prev_path = image.path or self.prev_path
        if field is not None:
            self.images_with_flow += 1
        preds = []
        for t in tracks:
            if field is not None:
                preds.append(np.asarray(field.warp_box(t.box), dtype=np.float64))
            else:
                dt = max(0.0, (image.timestamp - t.last_t) / 1e6)
                preds.append(t.box + t.vel * dt)
        pred = np.array(preds)

        if detections is None:
            self.images_without_detections += 1
            pairs: dict[int, tuple[int, float]] = {}
        else:
            pairs = self._match(tracks, pred, detections)

        for i, t in enumerate(tracks):
            dt = max(1e-3, (image.timestamp - t.last_t) / 1e6)
            t.steps += 1
            if detections is not None:
                t.data_steps += 1
            if i in pairs:
                j, agreement = pairs[i]
                det = detections[j]
                det_box = np.asarray(det.bbox, dtype=np.float64)
                new_box = cfg.smoothing * det_box + (1 - cfg.smoothing) * pred[i]
                # Lần khớp đầu tiên chỉ "bắt" vào box detector: box người vẽ ở keyframe có thể rộng/hẹp hơn
                # box detector, lấy hiệu hai box làm vận tốc sẽ ra chuyển động giả
                if t.matched_steps > 0:
                    measured = (new_box - t.box) / dt
                    t.vel = cfg.velocity_smoothing * measured + (1 - cfg.velocity_smoothing) * t.vel
                t.box = new_box
                t.misses = 0
                if t.matched_steps == 0:
                    t.first_det_label = det.label
                t.matched_steps += 1
                t.last_match, t.last_agreement = det, agreement
            else:
                if field is not None:
                    # Box dời theo chuyển động thật của ảnh: lấy luôn làm vận tốc cho ảnh sau (nếu ảnh sau không có flow)
                    t.vel = (pred[i] - t.box) / dt
                else:
                    t.vel = t.vel * COAST_VELOCITY_DAMPING
                t.box = pred[i]
                t.last_match, t.last_agreement = None, 0.0
                # Ảnh chưa có detection trong cache không phải bằng chứng object biến mất
                if detections is not None:
                    t.misses += 1
            t.last_t = image.timestamp
            if image.is_keyframe:
                t.hops += 1
            self._check_stop(t)

    def _match(
        self, tracks: list[Track], pred: np.ndarray, detections: list[Detection]
    ) -> dict[int, tuple[int, float]]:
        """Track "keep" được ghép trước; track "suppress" chỉ nhận detection còn thừa.

        Người hay xoá box trùng/lệch và giữ box đúng của cùng một object: nếu hai loại track tranh nhau thì
        track "suppress" có thể giành mất detection, làm nhãn thật bị coi là mất dấu.
        """
        pairs: dict[int, tuple[int, float]] = {}
        free = list(range(len(detections)))
        for kind in ("keep", "suppress"):
            idx = [i for i, t in enumerate(tracks) if t.kind == kind]
            if not idx or not free:
                continue
            found = _greedy_match(pred[idx], [detections[j].bbox for j in free], self.cfg.match_iou)
            for a, (b, v) in found.items():
                pairs[idx[a]] = (free[b], v)
            used = {free[b] for b, _ in found.values()}
            free = [j for j in free if j not in used]
        return pairs

    def _check_stop(self, t: Track) -> None:
        cfg = self.cfg
        if t.misses > cfg.max_coast_images:
            t.stop(f"Mất dấu: {t.misses} ảnh liên tiếp không có detection khớp")
            return
        b = t.box
        w, h = b[2] - b[0], b[3] - b[1]
        if w < cfg.min_box_px or h < cfg.min_box_px:
            t.stop("Box quá nhỏ")
            return
        visible = box_area(clip_box(b.tolist(), self.width, self.height)) / max(box_area(b.tolist()), 1e-9)
        if visible < cfg.min_visible:
            t.stop("Đi ra khỏi khung hình")


# ---------------------------------------------------------------------------
# Độ tin cậy lan truyền
# ---------------------------------------------------------------------------


def prop_confidence(
    agreement: float, continuity: float | None, lidar_ratio: float | None, hops: int, cfg: PropagationCfg
) -> float:
    """c_prop = trung bình có trọng số (bỏ thành phần không đo được) × hop_decay^(hops - 1)."""
    w = cfg.weights
    terms = [(w.agreement, agreement)]
    if continuity is not None:
        terms.append((w.continuity, continuity))
    if lidar_ratio is not None:
        terms.append((w.lidar, min(1.0, lidar_ratio)))
    total = sum(wt for wt, _ in terms)
    value = sum(wt * v for wt, v in terms) / total if total > 0 else 0.0
    return round(value * cfg.hop_decay ** max(0, hops - 1), 4)


def track_confidence(t: Track, uv: np.ndarray | None, cfg: PropagationCfg) -> float:
    lidar_ratio = None
    if t.ref_points:
        n = count_points(uv, t.output_box())
        lidar_ratio = None if n is None else n / t.ref_points
    return prop_confidence(t.last_agreement, t.continuity, lidar_ratio, t.hops, cfg)


# ---------------------------------------------------------------------------
# Ghi kết quả vào keyframe đích
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    propagated: int = 0
    suppressed: int = 0
    stopped: list[str] = field(default_factory=list)


def _propagated_qa(
    base: LabelObject | None,
    bbox: list[float],
    label: str,
    prop_conf: float,
    track: Track,
    frame: FrameRecord,
    uv: np.ndarray | None,
    depth: np.ndarray | None,
    config: AutoLabelConfig,
) -> QAResult:
    cfg = config.propagation
    base_qa = base.qa if base is not None else None
    width, height = frame.image.width, frame.image.height
    fy = frame.intrinsic[1][1]
    issues: list[QAIssue] = []
    terms: dict[str, float] = {}

    # LiDAR và hình học phụ thuộc lớp: tính lại khi lớp người chốt khác lớp detector hoặc box là box dự đoán
    recompute = base_qa is None or base.label != label
    if recompute:
        lid = check_lidar(bbox, label, uv, depth, fy, height, config)
        geo = check_geometry(LabelObject(object_id="_", bbox=bbox, label=label, score=0.0), width, height, config)
        issues += lid["issues"] + geo["issues"]
        terms["lidar"], terms["geometric"] = lid["term"], geo["term"]
        lidar_info = {k: v for k, v in lid.items() if k not in ("issues", "term")}
    else:
        issues += [i for i in base_qa.issues if i.group in ("lidar", "geometric")]
        terms["lidar"] = base_qa.terms.get("lidar", 0.0)
        terms["geometric"] = base_qa.terms.get("geometric", 0.0)
        lidar_info = base_qa.lidar

    temporal_term = 0.0
    temporal_info: dict = {}
    if base_qa is not None:
        issues += [i for i in base_qa.issues if i.group == "temporal" and i.code != "RECOVERED_BY_TRACK"]
        temporal_term = base_qa.terms.get("temporal", 0.0)
        temporal_info = base_qa.temporal
    if track.last_match is None:
        issues.append(
            QAIssue(
                code="PROP_COASTING",
                group="temporal",
                message=f"Detector không thấy object ở frame này; box dự đoán từ chuyển động ({track.misses} ảnh không khớp)",
            )
        )
        temporal_term = max(temporal_term, 0.8)
    terms["temporal"] = temporal_term

    detection_term = 1.0 - prop_conf
    det = track.last_match
    if det is not None and det.label not in (label, track.first_det_label) and det.score >= cfg.class_differs_score:
        issues.append(
            QAIssue(
                code="PROP_CLASS_DIFFERS",
                group="detection",
                message=f"Detector ở frame này nhận là '{det.label}' (score {det.score:.2f}), lớp lan truyền là '{label}'",
            )
        )
        detection_term += config.qa.risk.class_conflict_penalty
    if prop_conf < cfg.flag_below:
        issues.append(
            QAIssue(
                code="PROP_LOW_CONF",
                group="temporal",
                message=f"Độ tin cậy lan truyền {prop_conf:.2f} < {cfg.flag_below:.2f} (từ {track.keyframe_id})",
            )
        )
    terms["detection"] = min(1.0, detection_term)

    qa = QAResult(
        risk=0.0,
        level="low",
        issues=[i for i in issues if i.code not in _DETECTOR_ISSUES],
        terms={k: round(v, 3) for k, v in terms.items()},
        lidar=lidar_info,
        temporal=temporal_info,
    )
    return score_risk(qa, config.qa.risk)


def _claim(objects: list[LabelObject], claimed: set[int], box: list[float]) -> int | None:
    """Box pre-label (máy sinh, chưa bị track nào nhận) trùng box lan truyền nhất."""
    best, best_i = CLAIM_IOU, None
    for i, o in enumerate(objects):
        if i in claimed or o.source not in ("model", "track"):
            continue
        v = iou(o.bbox, box)
        if v >= best:
            best, best_i = v, i
    return best_i


def apply_propagation(
    frame: FrameRecord,
    keyframe_id: str,
    tracks: list[Track],
    config: AutoLabelConfig,
    lidar_aux: dict | None,
    at: str,
) -> ApplyResult:
    """Ghi các track còn sống vào keyframe đích (sửa `frame` tại chỗ). Chỉ gọi cho frame status "auto".

    Luôn bắt đầu từ bản pre-label (frame.prelabel), nên lan truyền lại từ keyframe khác không bị cộng dồn.
    Track có c_prop dưới stop_below bị dừng và không được ghi.
    """
    if frame.status != "auto":
        raise ValueError(f"Không lan truyền vào frame {frame.frame_id} đang ở trạng thái {frame.status}")
    cfg = config.propagation
    uv, depth = lidar_arrays(lidar_aux)
    if frame.prelabel is None:
        frame.prelabel = [o.model_copy(deep=True) for o in frame.objects]
    objects = [o.model_copy(deep=True) for o in frame.prelabel]
    result = ApplyResult()

    scored = []
    for t in tracks:
        if not t.alive:
            continue
        c = track_confidence(t, uv, cfg)
        if c < cfg.stop_below:
            t.stop(f"Độ tin cậy lan truyền {c:.2f} < {cfg.stop_below:.2f}")
            result.stopped.append(t.track_id)
            continue
        scored.append((c, t))
    # Track chắc hơn được nhận box pre-label trước
    scored.sort(key=lambda ct: (ct[1].kind != "keep", -ct[0]))

    claimed: set[int] = set()
    extra: list[LabelObject] = []
    for c, t in scored:
        box = t.output_box()
        idx = _claim(objects, claimed, box)
        info = PropagationInfo(
            keyframe_id=t.keyframe_id,
            keyframe_object_id=t.keyframe_object_id,
            prop_conf=c,
            matched=t.last_match is not None,
            steps=t.steps,
            detector_label=t.last_match.label if t.last_match is not None else None,
            detector_score=t.last_match.score if t.last_match is not None else None,
        )
        if t.kind == "suppress":
            # Chỉ tự xoá khi detector thật sự thấy lại đúng object đó: khớp chặt, cùng lớp
            if idx is None or t.last_match is None or t.last_agreement < cfg.suppress_iou:
                continue
            base = objects[idx]
            if base.label != t.label:
                continue
            claimed.add(idx)
            objects[idx] = base.model_copy(
                update={
                    "track_id": t.track_id,
                    "propagation": info,
                    "review": ReviewState(
                        status="deleted", action="PROPAGATED_DELETE", reviewer=PROPAGATION_REVIEWER, at=at
                    ),
                }
            )
            result.suppressed += 1
            continue

        if t.last_match is None and idx is None and not cfg.emit_coasting:
            # Không ghi box thuần dự đoán (không detection nào đỡ); track vẫn sống để bắt lại vật ở ảnh sau
            continue
        base = objects[idx] if idx is not None else None
        if base is not None:
            bbox = list(base.bbox)
        else:
            bbox = [round(v, 1) for v in clip_box(box, frame.image.width, frame.image.height)]
        obj = LabelObject(
            object_id=base.object_id if base is not None else f"p{len(extra) + 1}",
            bbox=bbox,
            label=t.label,
            score=base.score if base is not None else c,
            source="propagated",
            models=base.models if base is not None else {},
            alternatives={},
            track=base.track if base is not None else {},
            track_id=t.track_id,
            propagation=info,
            qa=_propagated_qa(base, bbox, t.label, c, t, frame, uv, depth, config),
        )
        if idx is not None:
            claimed.add(idx)
            objects[idx] = obj
        else:
            extra.append(obj)
        result.propagated += 1

    frame.objects = objects + extra
    frame.frame_risk = max(
        (o.qa.risk for o in frame.objects if o.qa and o.review.status == "pending"),
        default=0.0,
    )
    frame.propagated_from = keyframe_id
    frame.propagated_at = at
    return result
