"""Đánh giá pre-label và QA Agent so với GT 2D chiếu từ box 3D nuScenes.

GT đóng vai "người sửa giả lập": object máy sinh cần sửa nếu không khớp GT cùng lớp
ở IoU >= 0.5. Không dùng ngưỡng 0.9 của M4 vì hộp bao chiếu từ 3D luôn rộng hơn
box sát vật thể, dùng 0.9 thì gần như mọi box đều "sai".
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.models.schemas import FrameRecord
from src.services.geometry import iou


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """AP = diện tích dưới đường precision-recall (nội suy all-point như VOC 2010+)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def _match(preds: list[dict], gts: list[dict], thr: float, class_aware: bool = True) -> tuple[list[str], set[int]]:
    """Ghép pred (đã sắp theo score giảm dần) với GT. Trả về trạng thái từng pred và tập GT đã ghép.

    Trạng thái: tp, fp, hoặc ignore (trùng GT bị bỏ qua vì khuất > 60% / không có điểm đo).
    """
    used: set[int] = set()
    status = []
    for p in preds:
        best, best_j = thr, -1
        for j, g in enumerate(gts):
            if j in used or g["ignore"] or (class_aware and g["label"] != p["label"]):
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            status.append("tp")
        elif any(g["ignore"] and iou(p["bbox"], g["bbox"]) >= thr for g in gts):
            status.append("ignore")
        else:
            status.append("fp")
    return status, used


def prelabel_objects(frame: FrameRecord) -> list:
    """Object pre-label của detector. Frame đã được lan truyền giữ bản gốc ở frame.prelabel."""
    return frame.prelabel if frame.prelabel is not None else frame.objects


def evaluate_map(frames: list[FrameRecord], gt: dict[str, list[dict]], thr: float) -> dict:
    per_class: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    n_gt: dict[str, int] = defaultdict(int)
    for f in frames:
        gts = gt.get(f.frame_id, [])
        for g in gts:
            if not g["ignore"]:
                n_gt[g["label"]] += 1
        preds = sorted(
            ({"bbox": o.bbox, "label": o.label, "score": o.score} for o in prelabel_objects(f) if o.source == "model"),
            key=lambda p: -p["score"],
        )
        for label in {p["label"] for p in preds}:
            cls_preds = [p for p in preds if p["label"] == label]
            status, _ = _match(cls_preds, gts, thr)
            per_class[label] += [
                (p["score"], s == "tp") for p, s in zip(cls_preds, status, strict=True) if s != "ignore"
            ]

    ap = {}
    for label in sorted(n_gt):
        items = sorted(per_class.get(label, []), key=lambda x: -x[0])
        ap[label] = {
            "ap": round(average_precision(np.array([t for _, t in items], dtype=float), n_gt[label]), 4),
            "n_gt": n_gt[label],
        }
    valid = [v["ap"] for v in ap.values()]
    return {
        "iou": thr,
        "map": round(float(np.mean(valid)), 4) if valid else None,
        "n_classes": len(valid),
        "per_class": ap,
    }


def evaluate_qa(frames: list[FrameRecord], gt: dict[str, list[dict]], thr: float = 0.5) -> dict:
    """Cờ của QA Agent có đúng chỗ không (flag recall/precision) và tracking bù được bao nhiêu FN."""
    by_level = {lv: {"n": 0, "needs_fix": 0} for lv in ("low", "medium", "high")}
    by_issue: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "needs_fix": 0})
    n_eval = n_fix = flagged = flagged_fix = 0
    fn_total = fn_recovered = proposals = proposals_hit = 0

    for f in frames:
        gts = gt.get(f.frame_id, [])
        objects = prelabel_objects(f)
        model_objs = sorted((o for o in objects if o.source == "model" and o.qa), key=lambda o: -o.score)
        status, _ = _match([{"bbox": o.bbox, "label": o.label} for o in model_objs], gts, thr)
        for o, s in zip(model_objs, status, strict=True):
            if s == "ignore":
                continue
            fix = s == "fp"
            is_flagged = o.qa.level != "low"
            n_eval += 1
            n_fix += fix
            flagged += is_flagged
            flagged_fix += is_flagged and fix
            by_level[o.qa.level]["n"] += 1
            by_level[o.qa.level]["needs_fix"] += fix
            for code in {i.code for i in o.qa.issues}:
                by_issue[code]["n"] += 1
                by_issue[code]["needs_fix"] += fix

        # FN: GT không box máy nào chạm tới (không xét lớp)
        _, hit = _match([{"bbox": o.bbox, "label": o.label} for o in model_objs], gts, thr, class_aware=False)
        missed = [g for j, g in enumerate(gts) if not g["ignore"] and j not in hit]
        fn_total += len(missed)
        recovered = [o for o in objects if o.source == "track"]
        proposals += len(recovered)
        _, rec_hit = _match([{"bbox": o.bbox, "label": o.label} for o in recovered], missed, thr, class_aware=False)
        fn_recovered += len(rec_hit)
        _, any_hit = _match([{"bbox": o.bbox, "label": o.label} for o in recovered], gts, thr, class_aware=False)
        proposals_hit += len(any_hit)

    def ratio(a, b):
        return round(a / b, 4) if b else None

    low = by_level["low"]
    return {
        "match_iou": thr,
        "objects_evaluated": n_eval,
        "objects_need_fix": n_fix,
        "flag_recall": ratio(flagged_fix, n_fix),
        "flag_precision": ratio(flagged_fix, flagged),
        "low_risk_share": ratio(low["n"], n_eval),
        "low_risk_error_rate": ratio(low["needs_fix"], low["n"]),
        "by_level": {lv: {**v, "error_rate": ratio(v["needs_fix"], v["n"])} for lv, v in by_level.items()},
        "by_issue": {c: {**v, "precision": ratio(v["needs_fix"], v["n"])} for c, v in sorted(by_issue.items())},
        "fn_total": fn_total,
        "fn_recovered_by_track": fn_recovered,
        "track_proposals": proposals,
        "track_proposal_precision": ratio(proposals_hit, proposals),
    }


def render_report(result: dict) -> str:
    qa = result["qa"]
    lines = [
        "# Đánh giá AutoLabel 2D + QA Agent",
        "",
        f"- Frame: {result['n_frames']} keyframe {result['camera']} (nuScenes {result['version']})",
        f"- Detector: {', '.join(result['detectors'])}",
        "- GT: hộp bao 8 đỉnh box 3D chiếu xuống ảnh; bỏ qua object visibility 0-40% hoặc không có điểm đo",
        "",
        "## Chất lượng pre-label (trước khi người duyệt)",
        "",
        "| Lớp | #GT | AP@0.5 | AP@0.7 |",
        "| :-- | --: | --: | --: |",
    ]
    m5, m7 = result["map50"], result["map70"]
    for label, v in m5["per_class"].items():
        lines.append(f"| {label} | {v['n_gt']} | {v['ap']:.3f} | {m7['per_class'][label]['ap']:.3f} |")
    lines += [
        f"| **mAP** ({m5['n_classes']} lớp) | | **{m5['map']:.3f}** | **{m7['map']:.3f}** |",
        "",
        "AP@0.7 thấp là bình thường: GT chiếu từ 3D rộng hơn box sát vật thể của detector.",
        "",
        "## QA Agent — cờ có đúng chỗ không",
        "",
        f"Object cần sửa = không khớp GT cùng lớp ở IoU ≥ {qa['match_iou']}. "
        f"Đánh giá {qa['objects_evaluated']} object máy sinh, {qa['objects_need_fix']} cần sửa.",
        "",
        "| Chỉ số | Giá trị | Ý nghĩa |",
        "| :-- | --: | :-- |",
        f"| Flag recall | {_fmt(qa['flag_recall'])} | % object cần sửa bị agent đưa vào nhóm medium/high |",
        f"| Flag precision | {_fmt(qa['flag_precision'])} | % object bị gắn cờ thật sự cần sửa |",
        f"| Tỉ lệ nhóm low | {_fmt(qa['low_risk_share'])} | Phần object duyệt được theo lô |",
        f"| Lỗi lọt trong nhóm low | {_fmt(qa['low_risk_error_rate'])} | Rủi ro khi bấm Approve all low-risk |",
        f"| FN được tracking bù | {qa['fn_recovered_by_track']}/{qa['fn_total']} | GT detector sót, RECOVERED_BY_TRACK đề xuất đúng |",
        f"| Precision đề xuất tracking | {_fmt(qa['track_proposal_precision'])} | {qa['track_proposals']} box đề xuất |",
        "",
        "### Theo nhóm risk",
        "",
        "| Nhóm | #object | Tỉ lệ cần sửa |",
        "| :-- | --: | --: |",
    ]
    for lv, v in qa["by_level"].items():
        lines.append(f"| {lv} | {v['n']} | {_fmt(v['error_rate'])} |")
    lines += ["", "### Theo issue code", "", "| Issue | #object | Precision |", "| :-- | --: | --: |"]
    for code, v in qa["by_issue"].items():
        lines.append(f"| `{code}` | {v['n']} | {_fmt(v['precision'])} |")
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"
