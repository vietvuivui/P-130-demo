"""Mang nhãn của keyframe trước sang keyframe đang gán nhãn (qa.temporal.carry_prev).

Mỗi keyframe vốn được gán nhãn độc lập, nên vật mà detector thấy rõ ở frame trước (0.34) rồi thấy mờ ở frame này (0.20,
dưới ngưỡng giữ) sẽ biến mất rồi hiện lại — người duyệt phải vẽ tay. Ở đây tracker lan truyền (optical flow + ghép kiểu
ByteTrack, src/services/propagation.py) đi từ nhãn máy của keyframe trước, qua các ảnh 12 Hz ở giữa, tới keyframe này;
vật theo được tới nơi và khớp một detection "yếu" ở keyframe này thì detection đó được giữ lại làm RECOVERED_BY_TRACK
(nét đứt, phải xác nhận). Không bao giờ bịa box: chỉ nâng box detector đã có.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection, FrameRecord, ImageInfo, LabelObject
from src.services.geometry import iou
from src.services.nuscenes_data import TimelineImage


def prev_objects(prev: FrameRecord, min_score: float) -> list[tuple[list[float], str, str, float]]:
    """(box, lớp, object_id, score) của các vật ở frame trước đáng mang theo: người đã giữ / vẽ, hoặc máy sinh còn
    pending với score đủ cao. Vật người xoá thì không."""
    out = []
    for o in prev.objects:
        r = o.review
        if r.status == "deleted":
            continue
        box, label = r.final_bbox or o.bbox, r.final_label or o.label
        if r.status == "approved" or o.source == "human" or o.score >= min_score:
            out.append((box, label, o.object_id, o.score))
    return out


def carry_from_prev(
    prev: FrameRecord,
    image: ImageInfo,
    sweeps: dict[int, ImageInfo],
    key_dets: list[Detection],
    sweep_dets: dict[int, list[Detection]],
    image_file: Callable[[str], Path],
    config: AutoLabelConfig,
    max_gap_s: float = 1.5,
) -> list[LabelObject]:
    """Trả về các detection yếu ở keyframe (dưới ngưỡng giữ) mà tracker từ frame trước theo được tới, dạng LabelObject
    source=track. key_dets / sweep_dets: detection đầy đủ (>= score_threshold) của keyframe và các sweep của nó."""
    from src.services.flow import FlowProvider
    from src.services.propagation import Track, Tracker

    tcfg = config.qa.temporal
    if prev is None or not prev.image.path or (image.timestamp - prev.image.timestamp) > max_gap_s * 1e6:
        return []
    seeds = prev_objects(prev, tcfg.carry_min_score)
    if not seeds:
        return []
    tracks = [
        Track(track_id=f"{prev.frame_id}:{oid}", label=label, box=np.asarray(box, dtype=np.float64), kind="keep",
              keyframe_id=prev.frame_id, keyframe_object_id=oid, last_t=prev.image.timestamp)  # fmt: skip
        for box, label, oid, _ in seeds
    ]
    # Ảnh giữa hai keyframe: sweep sau của frame trước, sweep trước của frame này, rồi keyframe này
    timeline: list[tuple[TimelineImage, list[Detection] | None]] = []
    for s in sorted(prev.sweeps, key=lambda s: s.offset):
        if s.offset > 0 and s.path and prev.image.timestamp < s.timestamp < image.timestamp:
            timeline.append((TimelineImage(s.sd_token, s.timestamp, None, s.path), list(s.detections) + list(s.weak)))
    for o in sorted(sweeps):
        s = sweeps[o]
        if o < 0 and s.path and prev.image.timestamp < s.timestamp < image.timestamp:
            timeline.append((TimelineImage(s.sd_token, s.timestamp, None, s.path), sweep_dets.get(o)))
    timeline.append((TimelineImage(image.sd_token, image.timestamp, image.sd_token, image.path), key_dets))
    timeline.sort(key=lambda x: x[0].timestamp)

    pcfg = config.propagation.model_copy(update={"flow": "always", "association": "byte", "oc_recover": False})
    motion = FlowProvider(image_file, pcfg.flow_scale) if pcfg.flow != "off" else None
    tracker = Tracker(tracks, pcfg, image.width, image.height, motion, prev.image.path)
    for im, dets in timeline:
        tracker.step(im, dets)

    det_cfg = config.detection
    out: list[LabelObject] = []
    score_of = {oid: sc for _, _, oid, sc in seeds}
    for t in tracker.alive:
        d = t.last_match
        if d is None or d.label != t.label or det_cfg.keep(d.label, d.score):
            continue  # không khớp ở keyframe, khác lớp, hoặc detector đã giữ box đó rồi
        if any(iou(d.bbox, o.bbox) >= tcfg.match_iou for o in out):
            continue
        out.append(
            LabelObject(
                object_id=f"c{len(out) + 1}",
                bbox=[round(v, 1) for v in d.bbox],
                label=d.label,
                score=d.score,
                det_score=d.det_score if d.det_score is not None else d.score,
                source="track",
                mask=d.mask,
                track={"prev": [round(v, 1) for v in t.box.tolist()]},
                carried_from=f"{prev.frame_id}#{t.keyframe_object_id} ({score_of.get(t.keyframe_object_id, 0):.2f})",
            )
        )
    return out
