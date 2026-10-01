"""Audit ngẫu nhiên: ước lượng tỉ lệ lỗi còn lọt qua phần duyệt theo lô, có khoảng tin cậy.

Review by exception cho phép bấm "Approve all low-risk": object nhóm low không ai xem riêng. Vì thế tỉ lệ sửa của
nhóm low trong /metrics luôn ≈ 0 theo định nghĩa (không ai xem thì không ai sửa) và flag recall bị thổi phồng —
hai số đó không nói được gì về lỗi thật còn nằm trong nhóm low.

Audit lấy mẫu ngẫu nhiên (seed được lưu, tái lập được):
- kind=object: object đã duyệt theo lô (BATCH_APPROVE) -> người xem lại từng cái: đúng / sai.
- kind=frame: frame đã approve -> người xem cả frame: có vật nào bị sót (FN) không.

Tỉ lệ lỗi ước lượng bằng khoảng Wilson 95%. Mẫu sai thì frame được mở lại: object quay về chờ duyệt kèm issue
AUDIT_FAILED (nhóm high), frame thiếu vật nhận finding AUDIT_MISSING_OBJECT phải xử lý trước khi approve lại.
"""

from __future__ import annotations

import math
import random
import secrets

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import AuditItem, FrameRecord, QAIssue, QAResult, QCFinding, ReviewState
from src.services.review import now_iso, reopen_frame
from src.services.store import WorkspaceStore


class AuditError(ValueError):
    def __init__(self, message: str, status: int = 409, code: str = "AUDIT_ERROR"):
        super().__init__(message)
        self.status, self.code = status, code


def wilson_interval(errors: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ errors/n (đúng cả khi 0 lỗi hoặc n nhỏ, khác xấp xỉ chuẩn)."""
    if n == 0:
        return 0.0, 1.0
    p = errors / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _batch_approved(frames: list[FrameRecord]) -> list[tuple[FrameRecord, str]]:
    """Object không ai xem riêng: đã duyệt theo lô, trong frame đã approve."""
    return [
        (f, o.object_id)
        for f in frames
        if f.status == "approved"
        for o in f.objects
        if o.review.status == "approved" and o.review.action == "BATCH_APPROVE"
    ]


def create_sample(
    store: WorkspaceStore, config: AutoLabelConfig, kind: str, size: int | None = None, seed: int | None = None
) -> list[AuditItem]:
    items = store.load_audit()
    taken = {(i.kind, i.frame_id, i.object_id) for i in items}
    frames = store.list_frames()  # đã sắp theo frame_id: cùng seed + cùng dữ liệu -> cùng mẫu
    if kind == "object":
        pool = [(f, oid) for f, oid in _batch_approved(frames) if ("object", f.frame_id, oid) not in taken]
        size = size or config.qc.audit.object_sample_size
    else:
        pool = [(f, None) for f in frames if f.status == "approved" and ("frame", f.frame_id, None) not in taken]
        size = size or config.qc.audit.frame_sample_size
    if not pool:
        what = "object duyệt theo lô" if kind == "object" else "frame đã approve"
        raise AuditError(f"Không còn {what} nào chưa được lấy mẫu", code="AUDIT_EMPTY_POOL")
    if not size:
        raise AuditError("Cỡ mẫu phải ≥ 1", status=422)

    seed = seed if seed is not None else secrets.randbelow(2**31)
    picked = random.Random(seed).sample(pool, min(size, len(pool)))
    at = now_iso()
    new = []
    for n, (f, oid) in enumerate(picked, start=len(items) + 1):
        obj = next((o for o in f.objects if o.object_id == oid), None) if oid else None
        new.append(
            AuditItem(
                audit_id=f"a{n:05d}",
                kind=kind,
                frame_id=f.frame_id,
                object_id=oid,
                seed=seed,
                created_at=at,
                label=(obj.review.final_label or obj.label) if obj else None,
                bbox=(obj.review.final_bbox or obj.bbox) if obj else None,
            )
        )
    store.save_audit(items + new)
    store.append_qc_events(
        [{"event": "AUDIT_SAMPLE", "kind": kind, "seed": seed, "n": len(new), "pool": len(pool), "timestamp": at}]
    )
    return new


def _send_back(store: WorkspaceStore, config: AutoLabelConfig, item: AuditItem) -> None:
    """Mẫu sai: mở lại frame để người sửa bằng đúng luồng review."""
    frame = store.load_frame(item.frame_id)
    if frame is None:
        return
    if frame.status == "approved":
        reopen_frame(frame)
    reason = item.note or ("nhãn duyệt theo lô bị sai" if item.kind == "object" else "frame còn vật chưa gán nhãn")
    if item.kind == "object":
        obj = next((o for o in frame.objects if o.object_id == item.object_id), None)
        if obj is not None and obj.review.status != "deleted":
            qa = obj.qa or QAResult(risk=0.0, level="low")
            high = config.qa.risk.levels.high
            issue = QAIssue(code="AUDIT_FAILED", group="qc", message=f"Audit ngẫu nhiên ({item.audit_id}): {reason}")
            obj.qa = qa.model_copy(update={"issues": [*qa.issues, issue], "risk": max(qa.risk, high), "level": "high"})
            obj.review = ReviewState()
            frame.frame_risk = max(frame.frame_risk, obj.qa.risk)
    else:
        frame.qc_manual.append(
            QCFinding(
                code="AUDIT_MISSING_OBJECT",
                severity="warning",
                message=f"Audit ngẫu nhiên ({item.audit_id}): {reason}. Vẽ thêm box rồi xác nhận",
                frame_id=frame.frame_id,
                key=f"AUDIT_MISSING_OBJECT:{item.audit_id}",
                fingerprint="audit",
            )
        )
    store.save_frame(frame)


def record_result(
    store: WorkspaceStore, config: AutoLabelConfig, audit_id: str, result: str, note: str, reviewer: str
) -> AuditItem:
    items = store.load_audit()
    item = next((i for i in items if i.audit_id == audit_id), None)
    if item is None:
        raise AuditError(f"Không có mẫu audit {audit_id}", status=404, code="AUDIT_NOT_FOUND")
    if item.result is not None:
        raise AuditError(f"Mẫu {audit_id} đã có kết quả ({item.result})", code="AUDIT_DONE")
    item.result, item.note, item.reviewer, item.at = result, note.strip(), reviewer, now_iso()
    if result == "error":
        _send_back(store, config, item)
    store.save_audit(items)
    store.append_qc_events(
        [
            {
                "event": "AUDIT_RESULT",
                "audit_id": item.audit_id,
                "kind": item.kind,
                "frame_id": item.frame_id,
                "object_id": item.object_id,
                "result": result,
                "note": item.note,
                "reviewer": reviewer,
                "timestamp": item.at,
            }
        ]
    )
    return item


def _kind_summary(
    items: list[AuditItem], kind: str, population: set, verified: set, sample_size: int, max_upper: float, z: float
) -> dict:
    done = [i for i in items if i.kind == kind and i.result is not None]
    n, errors = len(done), sum(i.result == "error" for i in done)
    lo, hi = wilson_interval(errors, n, z)
    unaudited = len(population - verified)
    required = sample_size > 0
    census = bool(population) and unaudited == 0
    if not required:
        passed, note = True, "Không bắt buộc (sample_size = 0)"
    elif not population:
        passed, note = True, "Không có gì để audit"
    elif census:
        passed, note = True, "Đã audit toàn bộ" + (", mẫu sai đã được gửi lại sửa" if errors else "")
    elif n < sample_size:
        passed, note = False, f"Cần ít nhất {sample_size} mẫu có kết quả (đang có {n})"
    else:
        passed = hi <= max_upper
        note = f"Cận trên 95% {hi:.1%} {'≤' if passed else '>'} ngưỡng {max_upper:.0%}"
    return {
        "kind": kind,
        "required": required,
        "population": len(population),
        "unaudited": unaudited,
        "pending": sum(1 for i in items if i.kind == kind and i.result is None),
        "n": n,
        "errors": errors,
        "error_rate": round(errors / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "sample_size": sample_size,
        "max_error_upper": max_upper,
        "census": census,
        "passed": passed,
        "note": note,
    }


def audit_summary(
    store: WorkspaceStore, config: AutoLabelConfig, frames: list[FrameRecord] | None = None
) -> dict[str, dict]:
    cfg = config.qc.audit
    items = store.load_audit()
    frames = store.list_frames() if frames is None else frames
    ok = {(i.kind, i.frame_id, i.object_id) for i in items if i.result == "ok"}
    objects = {(f.frame_id, oid) for f, oid in _batch_approved(frames)}
    approved = {f.frame_id for f in frames if f.status == "approved"}
    return {
        "object": _kind_summary(
            items,
            "object",
            objects,
            {(fid, oid) for k, fid, oid in ok if k == "object"},
            cfg.object_sample_size,
            cfg.object_max_error_upper,
            cfg.confidence_z,
        ),
        "frame": _kind_summary(
            items,
            "frame",
            approved,
            {fid for k, fid, _ in ok if k == "frame"},
            cfg.frame_sample_size,
            cfg.frame_max_error_upper,
            cfg.confidence_z,
        ),
    }
