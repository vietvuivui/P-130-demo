"""Review by exception: áp thao tác của người lên nhãn, sinh correction log, tính số liệu."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, FrameSummary, LabelObject, ReviewActionRequest, ReviewState
from src.services.geometry import clip_box, iou

# PRD M4: object tính là "phải sửa" nếu bị xoá, đổi lớp, hoặc IoU với box gốc < 0.9
M4_IOU = 0.9


class ReviewError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log_entry(frame: FrameRecord, obj: LabelObject, action: str, reviewer: str) -> dict:
    return {
        "frame_id": frame.frame_id,
        "object_id": obj.object_id,
        "prediction": {"bbox": obj.bbox, "class": obj.label, "score": obj.score, "source": obj.source},
        "qa": {
            "risk": obj.qa.risk if obj.qa else None,
            "level": obj.qa.level if obj.qa else None,
            "issues": [i.code for i in obj.qa.issues] if obj.qa else [],
        },
        "human_action": action,
        "final_class": obj.review.final_label,
        "final_bbox": obj.review.final_bbox,
        "reviewer": reviewer,
        "timestamp": obj.review.at,
    }


def _find(frame: FrameRecord, object_id: str | None) -> LabelObject:
    for obj in frame.objects:
        if obj.object_id == object_id:
            return obj
    raise ReviewError(f"Không có object {object_id} trong frame {frame.frame_id}")


def _mark_editing(frame: FrameRecord) -> None:
    if frame.status == "approved":
        raise ReviewError("Frame đã approve, mở lại (reopen) trước khi sửa")
    frame.status = "editing"


def apply_action(frame: FrameRecord, req: ReviewActionRequest, reviewer: str, config: AutoLabelConfig) -> dict:
    """Áp một thao tác lên frame (sửa tại chỗ) và trả về dòng correction log."""
    _mark_editing(frame)
    if req.label is not None and req.label not in config.classes:
        raise ReviewError(f"Lớp không có trong taxonomy: {req.label}")
    w, h = frame.image.width, frame.image.height
    bbox = [round(v, 1) for v in clip_box(req.bbox, w, h)] if req.bbox else None
    if bbox and (bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2):
        raise ReviewError("Box quá nhỏ hoặc nằm ngoài ảnh")

    if req.action == "ADD_BOX":
        if bbox is None or req.label is None:
            raise ReviewError("ADD_BOX cần bbox và label")
        n = sum(o.source == "human" for o in frame.objects) + 1
        obj = LabelObject(object_id=f"h{n}", bbox=bbox, label=req.label, score=1.0, source="human",
                          mask=[round(v, 1) for v in req.mask] if req.mask else None)  # fmt: skip
        frame.objects.append(obj)
    else:
        obj = _find(frame, req.object_id)

    cur_label = obj.review.final_label or obj.label
    cur_bbox = obj.review.final_bbox or obj.bbox
    if req.action == "DELETE":
        obj.review = ReviewState(status="deleted", action="DELETE")
    elif req.action == "KEEP":
        obj.review = ReviewState(status="approved", action="KEEP", final_label=cur_label, final_bbox=cur_bbox)
    elif req.action == "CHANGE_CLASS":
        if req.label is None:
            raise ReviewError("CHANGE_CLASS cần label")
        obj.review = ReviewState(status="approved", action="CHANGE_CLASS", final_label=req.label, final_bbox=cur_bbox)
    elif req.action == "EDIT_BOX":
        if bbox is None:
            raise ReviewError("EDIT_BOX cần bbox")
        obj.review = ReviewState(
            status="approved", action="EDIT_BOX", final_label=req.label or cur_label, final_bbox=bbox
        )
    elif req.action == "ADD_BOX":
        obj.review = ReviewState(status="approved", action="ADD_BOX", final_label=req.label, final_bbox=bbox)

    obj.review.reviewer, obj.review.at = reviewer, now_iso()
    return _log_entry(frame, obj, req.action, reviewer)


def approve_low_risk(frame: FrameRecord, reviewer: str) -> list[dict]:
    """Duyệt theo lô mọi object rủi ro thấp còn chờ."""
    _mark_editing(frame)
    entries, at = [], now_iso()
    for obj in frame.objects:
        if obj.review.status == "pending" and obj.qa and obj.qa.level == "low":
            obj.review = ReviewState(
                status="approved",
                action="BATCH_APPROVE",
                final_label=obj.label,
                final_bbox=obj.bbox,
                reviewer=reviewer,
                at=at,
            )
            entries.append(_log_entry(frame, obj, "BATCH_APPROVE", reviewer))
    return entries


def approve_frame(frame: FrameRecord, reviewer: str, review_time_s: float | None) -> None:
    pending = [o.object_id for o in frame.objects if o.review.status == "pending"]
    if pending:
        raise ReviewError(f"Còn {len(pending)} object chưa duyệt: {', '.join(pending[:10])}")
    frame.status = "approved"
    frame.approved_at, frame.approved_by = now_iso(), reviewer
    frame.review_time_s = review_time_s


def reopen_frame(frame: FrameRecord) -> None:
    frame.status = "editing"
    frame.approved_at = frame.approved_by = None


def check_reason(reason: str) -> str:
    reason = " ".join((reason or "").split())
    if len(reason) < 3:
        raise ReviewError("Reject cần ghi lý do (ít nhất 3 ký tự)")
    return reason


def reject_frame(frame, reviewer: str, reason: str) -> None:
    """Reviewer trả lại frame kèm lý do (FR-15). Dùng chung cho frame 2D và 3D (cùng tên trường).

    Frame về trạng thái "rejected": không được xuất; sửa lại thì thành "editing" nhưng vẫn giữ lý do cho tới lần reject
    sau, để người gán nhãn thấy cần sửa gì. Duyệt lại bình thường bằng approve.
    """
    frame.reject_reason = check_reason(reason)
    frame.status, frame.rejected_by, frame.rejected_at = "rejected", reviewer, now_iso()
    frame.approved_at = frame.approved_by = None


def summarize(frame: FrameRecord) -> FrameSummary:
    counts = Counter(o.qa.level for o in frame.objects if o.qa)
    return FrameSummary(
        frame_id=frame.frame_id,
        scene=frame.scene,
        index=frame.index,
        status=frame.status,
        frame_risk=frame.frame_risk,
        counts={lv: counts.get(lv, 0) for lv in ("low", "medium", "high")},
        pending=sum(o.review.status == "pending" for o in frame.objects),
        n_objects=len(frame.objects),
        propagated_from=frame.propagated_from,
    )


def needs_fix(obj: LabelObject) -> bool:
    """Object máy sinh có bị người sửa không (định nghĩa M4)."""
    r = obj.review
    if r.status == "deleted":
        return True
    if r.final_label and r.final_label != obj.label:
        return True
    return bool(r.final_bbox) and iou(r.final_bbox, obj.bbox) < M4_IOU


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def compute_metrics(frames: list[FrameRecord]) -> dict:
    """Số liệu từ log thật: M4, M1, chất lượng cờ của QA Agent, hiệu quả tracking và lan truyền.

    m4_correction_rate chỉ tính object detector sinh (source "model"); nhãn lan truyền có M4 riêng
    trong "propagation". Cờ (flag_*) tính trên cả hai vì cả hai đều đi qua triage theo risk.
    """
    status = Counter(f.status for f in frames)  # auto / editing / approved / rejected
    levels, issue_counts = Counter(), Counter()
    model_reviewed, model_fixed = 0, 0
    flagged, flagged_fixed = 0, 0
    per_level = {lv: {"reviewed": 0, "fixed": 0} for lv in ("low", "medium", "high")}
    per_issue: dict[str, dict[str, int]] = {}
    track = {"proposed": 0, "accepted": 0, "rejected": 0}
    human_added = 0
    # Nhãn lan truyền: M4 riêng theo nguồn (PRD), và số object tự xoá theo keyframe
    prop = {"reviewed": 0, "fixed": 0, "pending": 0, "auto_suppressed": 0, "suppress_undone": 0}

    for f in frames:
        for o in f.objects:
            if o.qa:
                levels[o.qa.level] += 1
                issue_counts.update(i.code for i in o.qa.issues)
            if o.source == "human":
                human_added += 1
                continue
            if o.source == "track":
                track["proposed"] += 1
                if o.review.status == "approved":
                    track["accepted"] += 1
                elif o.review.status == "deleted":
                    track["rejected"] += 1
                continue
            if o.review.action == "PROPAGATED_DELETE":
                # Máy tự xoá theo keyframe, không phải người sửa: không tính vào M4 hay cờ
                prop["auto_suppressed"] += 1
                continue
            if o.propagation is not None and o.source != "propagated" and o.review.status == "approved":
                prop["suppress_undone"] += 1
            if o.source == "propagated" and o.review.status == "pending":
                prop["pending"] += 1
            if o.review.status == "pending" or o.qa is None:
                continue
            fixed = needs_fix(o)
            if o.source == "propagated":
                prop["reviewed"] += 1
                prop["fixed"] += fixed
            else:
                model_reviewed += 1
                model_fixed += fixed
            per_level[o.qa.level]["reviewed"] += 1
            per_level[o.qa.level]["fixed"] += fixed
            if o.qa.level != "low":
                flagged += 1
                flagged_fixed += fixed
            for code in {i.code for i in o.qa.issues}:
                s = per_issue.setdefault(code, {"flagged": 0, "fixed": 0})
                s["flagged"] += 1
                s["fixed"] += fixed

    times = [f.review_time_s for f in frames if f.status == "approved" and f.review_time_s]
    return {
        "frames": {"total": len(frames), **{k: status.get(k, 0) for k in ("auto", "editing", "approved", "rejected")}},
        "objects_by_level": {lv: levels.get(lv, 0) for lv in ("low", "medium", "high")},
        "issue_counts": dict(issue_counts.most_common()),
        "m4_correction_rate": _ratio(model_fixed, model_reviewed),
        "model_objects_reviewed": model_reviewed,
        "model_objects_fixed": model_fixed,
        # Cờ của agent so với việc người thực sự sửa
        "flag_precision": _ratio(flagged_fixed, flagged),
        "flag_recall": _ratio(flagged_fixed, model_fixed + prop["fixed"]),
        "fix_rate_by_level": {lv: {**v, "rate": _ratio(v["fixed"], v["reviewed"])} for lv, v in per_level.items()},
        "fix_rate_by_issue": {c: {**v, "rate": _ratio(v["fixed"], v["flagged"])} for c, v in sorted(per_issue.items())},
        "track_proposals": track,
        "human_added_boxes": human_added,
        "propagation": {**prop, "m4_correction_rate": _ratio(prop["fixed"], prop["reviewed"])},
        "m1_avg_review_time_s": round(sum(times) / len(times), 1) if times else None,
    }
