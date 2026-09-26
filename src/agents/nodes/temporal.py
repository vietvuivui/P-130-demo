"""3.3 Temporal consistency: so keyframe với các sweep camera 12Hz lân cận.

- FLICKER: box ở keyframe gần như không xuất hiện lại ở sweep lân cận -> dễ là FP.
- RECOVERED_BY_TRACK: object có ở sweep trước và sau nhưng detector sót ở keyframe
  -> đề xuất box nội suy theo thời gian (bắt FN, điểm yếu lớn nhất của pre-label).
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
    objects: list[LabelObject], sweeps: dict[int, SweepDetections], key_timestamp: int, config: AutoLabelConfig
) -> tuple[dict[str, dict], list[LabelObject]]:
    cfg = config.qa.temporal
    boxes = [o.bbox for o in objects]
    results = {o.object_id: {"presence": {}, "track": {}, "issues": [], "term": 0.0} for o in objects}
    unmatched: dict[int, list[Detection]] = {}

    for offset, sweep in sorted(sweeps.items()):
        dets = sweep["detections"]
        pairs = match_greedy(boxes, [d.bbox for d in dets], cfg.match_iou)
        for i, o in enumerate(objects):
            j = pairs.get(i)
            results[o.object_id]["presence"][str(offset)] = j is not None
            results[o.object_id]["track"][str(offset)] = dets[j].bbox if j is not None else None
        matched = set(pairs.values())
        unmatched[offset] = [d for j, d in enumerate(dets) if j not in matched]

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
    for obj in recovered:
        results[obj.object_id] = {
            "presence": {k: v is not None for k, v in obj.track.items()},
            "track": obj.track,
            "support": sum(v is not None for v in obj.track.values()),
            "available": available,
            "issues": [
                QAIssue(
                    code="RECOVERED_BY_TRACK",
                    group="temporal",
                    message="Detector sót ở keyframe; box nội suy từ sweep "
                    + " và ".join(f"t{int(k):+d}" for k in obj.track),
                )
            ],
            "term": 0.8,
        }
    return results, recovered


def _recover_missed(
    objects: list[LabelObject],
    sweeps: dict[int, SweepDetections],
    unmatched: dict[int, list[Detection]],
    key_timestamp: int,
    config: AutoLabelConfig,
) -> list[LabelObject]:
    cfg = config.qa.temporal
    before = [o for o in (-1, -2) if o in sweeps]
    after = [o for o in (1, 2) if o in sweeps]
    existing = [o.bbox for o in objects]
    proposals: list[LabelObject] = []

    for ob in before:
        for p in unmatched[ob]:
            if p.score < cfg.recover_min_score:
                continue
            best: tuple[float, int, Detection] | None = None
            for oa in after:
                for q in unmatched[oa]:
                    if q.score < cfg.recover_min_score or q.label != p.label:
                        continue
                    v = iou(p.bbox, q.bbox)
                    if v >= cfg.match_iou and (best is None or v > best[0]):
                        best = (v, oa, q)
            if best is None:
                continue
            _, oa, q = best
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
        state.get("objects", []), state.get("sweeps", {}), state.get("key_timestamp", 0), state["config"]
    )
    return {"temporal": results, "recovered": recovered}
