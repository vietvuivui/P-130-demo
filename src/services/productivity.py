"""Năng suất theo người / theo phiên và throughput inference (FR-27), báo cáo CSV (FR-19).

- Người: mỗi frame đã approve tính cho người approve nó; thời gian = thời gian duyệt UI đo (review_time_s).
  frame/giờ = số frame / tổng thời gian duyệt. Phiên = chuỗi lần approve của cùng người, cách nhau không quá SESSION_GAP.
- Inference: mỗi lần chạy auto-label (lệnh `run`, xử lý video, bước của dự án) có một mã phiên; frame/giờ = số frame /
  tổng thời gian auto-label của các frame trong phiên (đo khi xử lý, gồm detect + QA Agent).
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import UTC, datetime, timedelta

SESSION_GAP = timedelta(minutes=30)


def new_run_id(kind: str) -> str:
    """Mã một phiên chạy auto-label: loại + thời điểm bắt đầu + thiết bị."""
    try:
        from src.services.detectors.base import pick_device

        device = pick_device()
    except Exception:  # không có torch (detector demo / CI)
        device = "cpu"
    return f"{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{device}"


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _per_hour(n: int, seconds: float) -> float | None:
    return round(n * 3600 / seconds, 1) if seconds > 0 else None


def reviewer_stats(frames: list, events: list[dict], corrections: list[dict], mode: str) -> list[dict]:
    """frames: frame 2D hoặc 3D (cùng trường approved_by / approved_at / review_time_s)."""
    approvals = defaultdict(list)
    for f in frames:
        if f.status == "approved" and f.approved_by:
            approvals[f.approved_by].append((_ts(f.approved_at), f.review_time_s or 0.0, f))
    objects = defaultdict(int)
    for c in corrections:
        if c.get("reviewer") and c.get("human_action") not in ("UNDO", "REDO"):
            objects[c["reviewer"]] += 1
    rejects, undos = defaultdict(int), defaultdict(int)
    for e in events:
        if e.get("mode") != mode:
            continue
        if e.get("type") == "reject":
            rejects[e.get("reviewer")] += 1
        elif e.get("type") in ("undo", "redo"):
            undos[e.get("reviewer")] += 1
    out = []
    for who in sorted(set(approvals) | set(objects) | set(rejects)):
        items = sorted((a for a in approvals.get(who, []) if a[0]), key=lambda a: a[0])
        sessions, cur = [], []
        for a in items:
            if cur and a[0] - cur[-1][0] > SESSION_GAP:
                sessions.append(cur)
                cur = []
            cur.append(a)
        if cur:
            sessions.append(cur)
        secs = sum(a[1] for a in approvals.get(who, []))
        out.append({
            "reviewer": who,
            "frames_approved": len(approvals.get(who, [])),
            "objects_handled": objects.get(who, 0),
            "frames_rejected": rejects.get(who, 0),
            "undo_redo": undos.get(who, 0),
            "review_time_s": round(secs, 1),
            "frames_per_hour": _per_hour(len(approvals.get(who, [])), secs),
            "sessions": [
                {
                    "start": s[0][0].isoformat(timespec="seconds"),
                    "end": s[-1][0].isoformat(timespec="seconds"),
                    "frames": len(s),
                    "review_time_s": round(sum(a[1] for a in s), 1),
                    "frames_per_hour": _per_hour(len(s), sum(a[1] for a in s)),
                    # thời gian thực từ lúc bắt đầu frame đầu tới lúc approve frame cuối (gồm cả nghỉ ngắn)
                    "wall_frames_per_hour": _per_hour(len(s), (s[-1][0] - s[0][0]).total_seconds() + s[0][1]),
                }
                for s in sessions
            ],
        })  # fmt: skip
    return out


def inference_stats(frames: list) -> list[dict]:
    """Throughput auto-label theo phiên chạy (frame có autolabel_s / autolabel_run)."""
    runs = defaultdict(list)
    for f in frames:
        if f.autolabel_s is not None:
            runs[f.autolabel_run or "không rõ phiên"].append(f.autolabel_s)
    out = []
    for run, secs in sorted(runs.items()):
        total = sum(secs)
        out.append({
            "run": run,
            "device": run.rsplit("-", 1)[-1] if run.count("-") >= 2 else None,
            "frames": len(secs),
            "seconds": round(total, 1),
            "s_per_frame": round(total / len(secs), 3),
            "frames_per_hour": _per_hour(len(secs), total),
        })  # fmt: skip
    return out


def frames_csv(frames: list, mode: str) -> str:
    """Bảng từng frame: trạng thái, số object theo mức rủi ro, số đã sửa, người / thời gian duyệt, thời gian auto-label."""
    from src.services.review import needs_fix

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["mode", "frame_id", "scene", "index", "status", "objects", "high", "medium", "low", "pending",
                "fixed", "deleted", "human_added", "approved_by", "approved_at", "review_time_s", "rejected_by",
                "reject_reason", "autolabel_s", "autolabel_run"])  # fmt: skip
    for f in sorted(frames, key=lambda f: (f.scene, f.index)):
        levels = defaultdict(int)
        fixed = deleted = added = pending = 0
        for o in f.objects:
            lv = o.qa.level if getattr(o, "qa", None) else (o.verify.level if getattr(o, "verify", None) else None)
            if lv:
                levels[lv] += 1
            pending += o.review.status == "pending"
            deleted += o.review.status == "deleted"
            added += o.source == "human"
            if mode == "2d" and o.source != "human" and o.review.status != "pending":
                fixed += needs_fix(o)
            elif mode == "3d" and o.source != "human":
                fixed += o.review.action in ("DELETE", "CHANGE_CLASS", "EDIT_BOX")
        w.writerow([mode, f.frame_id, f.scene, f.index, f.status, len(f.objects), levels["high"], levels["medium"],
                    levels["low"], pending, fixed, deleted, added, f.approved_by or "", f.approved_at or "",
                    f.review_time_s if f.review_time_s is not None else "", f.rejected_by or "", f.reject_reason or "",
                    f.autolabel_s if f.autolabel_s is not None else "", f.autolabel_run or ""])  # fmt: skip
    return buf.getvalue()


def summary_csv(metrics: dict, prefix: str = "") -> str:
    """Số liệu tổng hợp (dict lồng nhau) thành bảng hai cột key,value."""
    rows = []

    def walk(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{key}[{i}]")
        else:
            rows.append((key, "" if obj is None else obj))

    walk(metrics, prefix)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["metric", "value"])
    w.writerows(rows)
    return buf.getvalue()
