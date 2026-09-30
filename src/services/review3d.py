"""Duyệt nhãn 3D theo ngoại lệ: Keep / Delete / đổi lớp từng box, duyệt lô nhóm rủi ro thấp, approve frame, số liệu."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from src.models.schemas import ReviewState
from src.models.schemas3d import Action3DRequest, Box3D, Frame3DRecord, Object3D
from src.services.geometry import quaternion_to_matrix
from src.services.label3d import now_iso
from src.services.store import WorkspaceStore

FIX_ACTIONS = {"DELETE", "CHANGE_CLASS", "EDIT_BOX"}
MIN_SIZE, MAX_SIZE = 0.1, 30.0  # m, giới hạn kích thước box người vẽ


class Review3DError(ValueError):
    pass


def _entry(frame: Frame3DRecord, obj: Object3D, action: str, reviewer: str) -> dict:
    v = obj.verify
    pred_box = obj.original_box or obj.box
    return {
        "dim": "3d",
        "model": frame.model,
        "frame_id": frame.frame_id,
        "object_id": obj.object_id,
        "source": obj.source,
        "prediction": None
        if obj.source == "human"
        else {"class": obj.label, "score": obj.score, **pred_box.model_dump()},
        "final_box": obj.box.model_dump() if action in ("EDIT_BOX", "ADD_BOX") else None,
        "verify": {"verdict": v.verdict, "level": v.level} if v else None,
        "human_action": action,
        "final_class": obj.review.final_label,
        "reviewer": reviewer,
        "timestamp": obj.review.at,
    }


def append_log(store: WorkspaceStore, entries: list[dict]) -> None:
    if not entries:
        return
    path = store.root / "corrections3d.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_log(store: WorkspaceStore, model: str | None = None) -> list[dict]:
    path = store.root / "corrections3d.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if model is None or r.get("model") == model]


def _find(frame: Frame3DRecord, object_id: str) -> Object3D:
    for o in frame.objects:
        if o.object_id == object_id:
            return o
    raise Review3DError(f"Không có box {object_id} trong frame {frame.frame_id}")


def _editable(frame: Frame3DRecord) -> None:
    if frame.status == "approved":
        raise Review3DError("Frame đã approve, mở lại trước khi sửa")
    frame.status = "editing"


def _check_box(req: Action3DRequest) -> Box3D:
    if req.box is None:
        raise Review3DError(f"{req.action} cần box")
    if not all(MIN_SIZE <= s <= MAX_SIZE for s in req.box.size):
        raise Review3DError(f"Kích thước box phải trong [{MIN_SIZE}, {MAX_SIZE}] m")
    return Box3D(
        center=[round(v, 3) for v in req.box.center], size=[round(v, 3) for v in req.box.size],
        yaw=round(math.remainder(req.box.yaw, 2 * math.pi), 5), velocity=req.box.velocity,
    )  # fmt: skip


def apply_action(frame: Frame3DRecord, req: Action3DRequest, reviewer: str, classes: list[str]) -> dict:
    _editable(frame)
    at = now_iso()
    if req.label is not None and req.label not in classes:
        raise Review3DError(f"Lớp không có trong taxonomy: {req.label}")
    if req.action == "ADD_BOX":
        # vật mô hình bỏ sót: người vẽ thêm, tự coi là đã duyệt
        if req.label is None:
            raise Review3DError("ADD_BOX cần label")
        oid = str(max((int(o.object_id) for o in frame.objects if o.object_id.isdigit()), default=0) + 1)
        obj = Object3D(
            object_id=oid, label=req.label, score=1.0, box=_check_box(req), source="human",
            review=ReviewState(status="approved", action="ADD_BOX", final_label=req.label, reviewer=reviewer, at=at),
        )  # fmt: skip
        frame.objects.append(obj)
        return _entry(frame, obj, "ADD_BOX", reviewer)
    if req.object_id is None:
        raise Review3DError(f"{req.action} cần object_id")
    obj = _find(frame, req.object_id)
    if req.action == "EDIT_BOX":
        box = _check_box(req)
        if obj.source == "model" and obj.original_box is None:
            obj.original_box = obj.box
        obj.box = box
        label = req.label or obj.review.final_label or obj.label
        action = "ADD_BOX" if obj.source == "human" else "EDIT_BOX"  # sửa lại box mình vẽ vẫn là box người thêm
        obj.review = ReviewState(status="approved", action=action, final_label=label, reviewer=reviewer, at=at)
        return _entry(frame, obj, "EDIT_BOX", reviewer)
    if req.action == "KEEP":
        obj.review = ReviewState(status="approved", action="KEEP", final_label=obj.label, reviewer=reviewer, at=at)
    elif req.action == "DELETE":
        obj.review = ReviewState(status="deleted", action="DELETE", reviewer=reviewer, at=at)
    else:
        if req.label is None:
            raise Review3DError("CHANGE_CLASS cần label")
        obj.review = ReviewState(
            status="approved", action="CHANGE_CLASS", final_label=req.label, reviewer=reviewer, at=at
        )
    return _entry(frame, obj, req.action, reviewer)


def approve_low_risk(frame: Frame3DRecord, reviewer: str) -> list[dict]:
    _editable(frame)
    at, out = now_iso(), []
    for o in frame.objects:
        if o.review.status == "pending" and o.verify is not None and o.verify.level == "low":
            o.review = ReviewState(
                status="approved", action="BATCH_APPROVE", final_label=o.label, reviewer=reviewer, at=at
            )
            out.append(_entry(frame, o, "BATCH_APPROVE", reviewer))
    return out


def approve_frame(frame: Frame3DRecord, reviewer: str, review_time_s: float | None) -> None:
    pending = [o.object_id for o in frame.objects if o.review.status == "pending"]
    if pending:
        raise Review3DError(f"Còn {len(pending)} box chưa duyệt: {', '.join(pending[:10])}")
    frame.status, frame.approved_at, frame.approved_by = "approved", now_iso(), reviewer
    frame.review_time_s = review_time_s


def reopen(frame: Frame3DRecord) -> None:
    frame.status, frame.approved_at, frame.approved_by = "editing", None, None


def metrics3d(frames: list[Frame3DRecord]) -> dict:
    """Tỉ lệ phải sửa theo kết luận / mức rủi ro, và độ chính xác của cờ (tính từ thao tác của người)."""
    by_level = {lv: {"reviewed": 0, "fixed": 0} for lv in ("low", "medium", "high")}
    by_verdict: dict[str, dict[str, int]] = {}
    flagged = flagged_fixed = fixed = reviewed = 0
    times = [f.review_time_s for f in frames if f.status == "approved" and f.review_time_s]
    for f in frames:
        for o in f.objects:
            if o.review.status == "pending" or o.verify is None:
                continue
            fix = o.review.action in FIX_ACTIONS
            reviewed += 1
            fixed += fix
            by_level[o.verify.level]["reviewed"] += 1
            by_level[o.verify.level]["fixed"] += fix
            d = by_verdict.setdefault(o.verify.verdict, {"reviewed": 0, "fixed": 0})
            d["reviewed"] += 1
            d["fixed"] += fix
            if o.verify.level != "low":
                flagged += 1
                flagged_fixed += fix
    return {
        "frames": len(frames),
        "approved": sum(f.status == "approved" for f in frames),
        "rejected": sum(f.status == "rejected" for f in frames),
        "objects": sum(o.source == "model" for f in frames for o in f.objects),
        "added": sum(o.source == "human" and o.review.status == "approved" for f in frames for o in f.objects),
        "edited": sum(o.review.action == "EDIT_BOX" for f in frames for o in f.objects),
        "reviewed": reviewed,
        "fixed": fixed,
        "correction_rate": round(fixed / reviewed, 4) if reviewed else None,
        "flag_precision": round(flagged_fixed / flagged, 4) if flagged else None,
        "flag_recall": round(flagged_fixed / fixed, 4) if fixed else None,
        "mean_review_s": round(float(np.mean(times)), 1) if times else None,
        "by_level": by_level,
        "by_verdict": by_verdict,
        "verdicts": dict(Counter(o.verify.verdict for f in frames for o in f.objects if o.verify)),
    }


def export3d(store: WorkspaceStore, model: str, out_dir: Path) -> dict:
    """Frame đã approve -> file kết quả chuẩn nuScenes (hệ toàn cục) với lớp người chốt, kèm log thao tác."""
    frames = [f for f in store.list_frames3d(model) if f.status == "approved"]
    if not frames:
        raise Review3DError("Chưa có frame 3D nào được approve")
    results = {}
    for f in frames:
        g = np.asarray(f.global_from_lidar)
        rows = []
        for o in f.objects:
            if o.review.status != "approved":
                continue
            c = g[:3, :3] @ np.asarray(o.box.center) + g[:3, 3]
            yaw_rot = np.array(
                [[np.cos(o.box.yaw), -np.sin(o.box.yaw), 0], [np.sin(o.box.yaw), np.cos(o.box.yaw), 0], [0, 0, 1]]
            )
            q = _matrix_to_quaternion(g[:3, :3] @ yaw_rot)
            v = g[:3, :3] @ np.array([*o.box.velocity, 0.0])
            rows.append({
                "sample_token": f.sample_token, "translation": c.round(3).tolist(), "size": o.box.size,
                "rotation": q, "velocity": v[:2].round(3).tolist(), "detection_name": o.review.final_label or o.label,
                "detection_score": o.score, "attribute_name": "", "review_action": o.review.action, "source": o.source,
                "verdict": o.verify.verdict if o.verify else None,
            })  # fmt: skip
        results[f.sample_token] = rows
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}
    (out_dir / "labels3d_nusc.json").write_text(json.dumps({"meta": meta, "results": results}))
    (out_dir / "corrections3d.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in read_log(store, model)), encoding="utf-8"
    )
    n = sum(len(r) for r in results.values())
    return {"model": model, "frames": len(frames), "objects": n, "dir": str(out_dir)}


def _matrix_to_quaternion(m: np.ndarray) -> list[float]:
    """Ma trận quay -> quaternion [w, x, y, z]."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    assert np.allclose(quaternion_to_matrix(q), m, atol=1e-4)
    return [round(float(x), 6) for x in q]
