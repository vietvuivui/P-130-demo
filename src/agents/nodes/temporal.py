"""3.3 Temporal consistency: so keyframe với các sweep camera 12Hz lân cận.

- FLICKER: box ở keyframe gần như không xuất hiện lại ở sweep lân cận -> dễ là FP.
- RECOVERED_BY_TRACK: object có ở sweep trước và sau nhưng detector sót ở keyframe
  -> đề xuất box nội suy theo thời gian (bắt FN, điểm yếu lớn nhất của pre-label). Nếu detector thật ra có thấy vật ở
  keyframe nhưng score dưới ngưỡng giữ (box "yếu", state["weak_key"]) thì dùng luôn box của detector (sát vật hơn box
  nội suy), và box yếu được sweep hai bên xác nhận (kể cả box sweep score thấp, sweep["weak"]) cũng được đề xuất —
  cùng ý với lượt ghép score thấp của ByteTrack: box mờ chỉ được tin khi đã có bằng chứng vật ở đó.
"""

from __future__ import annotations

import numpy as np

from src.agents.state import QAState, SweepDetections
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection, LabelObject, QAIssue
from src.services.geometry import interpolate_box, iou, iou_matrix


def match_greedy(a: list[list[float]], b: list[list[float]], thr: float) -> dict[int, int]:
    """Ghép cặp một-một theo IoU giảm dần. Trả về {chỉ số trong a: chỉ số trong b}."""
    m = iou_matrix(np.array(a), np.array(b))
    rows, cols = np.where(m >= thr)
    out: dict[int, int] = {}
    used_b: set[int] = set()
    for i, j in sorted(zip(rows.tolist(), cols.tolist(), strict=True), key=lambda ij: -m[ij]):
        if i not in out and j not in used_b:
            out[i] = j
            used_b.add(j)
    return out


def check_temporal(
    objects: list[LabelObject],
    sweeps: dict[int, SweepDetections],
    key_timestamp: int,
    config: AutoLabelConfig,
    weak_key: list[Detection] | None = None,
    carried: list[LabelObject] | None = None,
) -> tuple[dict[str, dict], list[LabelObject]]:
    cfg = config.qa.temporal
    boxes = [o.bbox for o in objects]
    results = {o.object_id: {"presence": {}, "track": {}, "issues": [], "term": 0.0} for o in objects}
    unmatched: dict[int, list[Detection]] = {}

    for offset, sweep in sorted(sweeps.items()):
        dets = sweep["detections"]
        # Box sweep đã dời về thời điểm keyframe bằng optical flow (nếu có), để so khớp không lệch vì vật đang chạy
        pairs = match_greedy(boxes, sweep.get("warped") or [d.bbox for d in dets], cfg.match_iou)
        for i, o in enumerate(objects):
            j = pairs.get(i)
            results[o.object_id]["presence"][str(offset)] = j is not None
            results[o.object_id]["track"][str(offset)] = dets[j].bbox if j is not None else None
        matched = set(pairs.values())
        warped = sweep.get("warped") or [None] * len(dets)
        unmatched[offset] = [(d, w) for j, (d, w) in enumerate(zip(dets, warped, strict=True)) if j not in matched]

    available = len(sweeps)
    for o in objects:
        r = results[o.object_id]
        support = sum(r["presence"].values())
        r["support"], r["available"] = support, available
        if available >= cfg.min_support and support < cfg.min_support:
            r["issues"].append(
                QAIssue(
                    code="FLICKER", group="temporal", message=f"Chỉ xuất hiện ở {support}/{available} sweep lân cận"
                )
            )
            r["term"] = 1.0
        elif available:
            r["term"] = 0.5 * (1 - support / available)

    recovered = _recover_missed(objects, sweeps, unmatched, key_timestamp, config)
    weak_used = _snap_to_weak(recovered, weak_key or [], config)
    recovered += _recover_weak(objects + recovered, sweeps, unmatched, weak_key or [], weak_used, config)
    for c in carried or []:  # vật tracker mang từ keyframe trước (carry.py); bỏ nếu đã có box trùng
        if not any(iou(c.bbox, o.bbox) >= cfg.match_iou and o.label == c.label for o in objects + recovered):
            recovered.append(c)
    for obj in recovered:
        at = " và ".join(f"t{int(k):+d}" for k in obj.track if k != "prev")
        lo = config.detection.min_score_per_class.get(obj.label, config.detection.min_score)
        if obj.carried_from:
            msg = (f"Detector thấy mờ ở keyframe (score {obj.score:.2f} < {lo:.2f}); tracker theo được vật "
                   f"{obj.carried_from} từ frame trước tới đây")
        elif obj.det_score is not None and obj.det_score > obj.score:  # score bị hạ khi gộp với box 3D (lidar2d.py)
            msg = (f"Detector thấy ở keyframe (score {obj.det_score:.2f}, còn {obj.score:.2f} sau khi gộp với box 3D, "
                   f"dưới ngưỡng {lo:.2f}); vật có ở sweep {at}")
        elif obj.det_score is not None:  # box của detector ở keyframe, score dưới ngưỡng giữ
            msg = f"Detector thấy mờ ở keyframe (score {obj.score:.2f} < {lo:.2f}); vật có ở sweep {at}"
        else:
            msg = f"Detector sót ở keyframe; box nội suy từ sweep {at}"
        results[obj.object_id] = {
            "presence": {k: v is not None for k, v in obj.track.items()},
            "track": obj.track,
            "support": sum(v is not None for v in obj.track.values()),
            "available": available,
            "issues": [QAIssue(code="RECOVERED_BY_TRACK", group="temporal", message=msg)],
            "term": 0.8,
        }
    return results, recovered


def _snap_to_weak(recovered: list[LabelObject], weak_key: list[Detection], config: AutoLabelConfig) -> set[int]:
    """Box nội suy trùng một detection keyframe bị bỏ vì score thấp -> lấy box của detector (sát vật hơn). Trả về chỉ
    số các detection yếu đã dùng."""
    used: set[int] = set()
    thr = config.qa.temporal.match_iou
    for r in recovered:
        best = max(
            ((iou(r.bbox, d.bbox), i) for i, d in enumerate(weak_key) if i not in used and d.label == r.label),
            default=(0.0, -1),
        )
        if best[0] >= thr:
            d = weak_key[best[1]]
            used.add(best[1])
            r.bbox, r.mask = [round(v, 1) for v in d.bbox], d.mask
            r.score, r.det_score = d.score, d.det_score if d.det_score is not None else d.score
    return used


def _recover_weak(
    taken: list[LabelObject],
    sweeps: dict[int, SweepDetections],
    unmatched: dict[int, list[tuple[Detection, list[float] | None]]],
    weak_key: list[Detection],
    used: set[int],
    config: AutoLabelConfig,
) -> list[LabelObject]:
    """Detection keyframe score thấp (bị bỏ) được sweep trước VÀ sau xác nhận (box cùng lớp, IoU >= match_iou, score >=
    recover_weak_min_score — kể cả box sweep chưa qua ngưỡng giữ) -> đề xuất box của detector."""
    cfg = config.qa.temporal
    lo = cfg.recover_weak_min_score
    if lo is None or not weak_key or not sweeps:
        return []
    before = [o for o in sweeps if o < 0]
    after = [o for o in sweeps if o > 0]
    if not before or not after:
        return []

    def evidence(d: Detection, offsets: list[int]) -> tuple[int, Detection] | None:
        best: tuple[float, int, Detection] | None = None
        for o in offsets:
            cands = [(p, pw) for p, pw in unmatched.get(o, [])] + [(p, None) for p in sweeps[o].get("weak", [])]
            for p, pw in cands:
                if p.label != d.label or p.score < lo:
                    continue
                v = iou(d.bbox, pw if pw is not None else p.bbox)
                if v >= cfg.match_iou and (best is None or v > best[0]):
                    best = (v, o, p)
        return (best[1], best[2]) if best else None

    out: list[LabelObject] = []
    occupied = [(o.bbox, o.label) for o in taken]

    def taken_already(d: Detection) -> bool:
        # Trùng box cùng lớp đã có, hoặc gần như cùng một box (người đi xe đạp: pedestrian + bicycle chồng nhau là hợp lệ)
        return any(iou(d.bbox, b) >= (cfg.match_iou if lb == d.label else 0.7) for b, lb in occupied)

    for i, d in enumerate(weak_key):
        if i in used or d.score < lo or taken_already(d):
            continue
        b, a = evidence(d, before), evidence(d, after)
        if b is None or a is None:
            continue
        out.append(
            LabelObject(
                object_id=f"w{len(out) + 1}",
                bbox=[round(v, 1) for v in d.bbox],
                label=d.label,
                score=d.score,
                det_score=d.det_score if d.det_score is not None else d.score,
                source="track",
                mask=d.mask,
                track={str(b[0]): b[1].bbox, str(a[0]): a[1].bbox},
            )
        )
        occupied.append((d.bbox, d.label))
    return out


def _recover_missed(
    objects: list[LabelObject],
    sweeps: dict[int, SweepDetections],
    unmatched: dict[int, list[tuple[Detection, list[float] | None]]],
    key_timestamp: int,
    config: AutoLabelConfig,
) -> list[LabelObject]:
    cfg = config.qa.temporal
    before = [o for o in (-1, -2) if o in sweeps]
    after = [o for o in (1, 2) if o in sweeps]
    existing = [o.bbox for o in objects]
    proposals: list[LabelObject] = []

    for ob in before:
        for p, pw in unmatched[ob]:
            if p.score < cfg.recover_min_score:
                continue
            best: tuple[float, int, Detection, list[float] | None] | None = None
            for oa in after:
                for q, qw in unmatched[oa]:
                    if q.score < cfg.recover_min_score or q.label != p.label:
                        continue
                    # Có flow: so hai box sau khi cùng dời về keyframe (vật chạy nhanh vẫn khớp)
                    v = iou(pw, qw) if pw is not None and qw is not None else iou(p.bbox, q.bbox)
                    if v >= cfg.match_iou and (best is None or v > best[0]):
                        best = (v, oa, q, qw)
            if best is None:
                continue
            _, oa, q, qw = best
            if pw is not None and qw is not None:
                box = [(a + b) / 2 for a, b in zip(pw, qw, strict=True)]
            else:
                box = interpolate_box(p.bbox, sweeps[ob]["timestamp"], q.bbox, sweeps[oa]["timestamp"], key_timestamp)
            if any(iou(box, b) >= cfg.match_iou for b in existing + [r.bbox for r in proposals]):
                continue
            proposals.append(
                LabelObject(
                    object_id=f"r{len(proposals) + 1}",
                    bbox=[round(v, 1) for v in box],
                    label=p.label,
                    score=round(min(p.score, q.score), 4),
                    source="track",
                    track={str(ob): p.bbox, str(oa): q.bbox},
                )
            )
    return proposals


def temporal_node(state: QAState) -> dict:
    results, recovered = check_temporal(
        state.get("objects", []), state.get("sweeps", {}), state.get("key_timestamp", 0), state["config"],
        state.get("weak_key"), state.get("carried"),
    )  # fmt: skip
    return {"temporal": results, "recovered": recovered}
