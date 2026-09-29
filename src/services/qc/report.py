"""QC toàn dataset: checklist "sẵn sàng phát hành" trước khi xuất, và báo cáo QA đi kèm bản export.

READY khi đúng hết:
1. Có frame đã approve.
2. Nhãn cuối của mọi frame đã approve không còn finding QC chưa sửa / chưa xác nhận.
3. Mỗi track (một vật, track_id xuyên frame) giữ một lớp duy nhất.
4. Correction log khớp nhãn cuối: object nào đã quyết định cũng có dòng log, và dòng cuối mô tả đúng trạng thái
   đang lưu (không có log nói đã sửa mà nhãn không đổi, cũng không có nhãn đổi mà không có log).
5. Audit ngẫu nhiên đạt (xem audit.py).

Frame chưa approve không chặn READY: bản export chỉ gồm frame đã approve; tỉ lệ phủ được ghi trong báo cáo.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, QCFinding
from src.services.qc.audit import audit_summary
from src.services.qc.checks import frame_findings, open_findings
from src.services.store import WorkspaceStore

# Sai số cho phép khi so box trong log với box đang lưu (box được làm tròn 0.1 px)
BOX_TOL = 0.05
# Máy tự làm, không đi qua thao tác của người nên không có dòng correction log
MACHINE_ACTIONS = {"PROPAGATED_DELETE"}


def load_frame_findings(store: WorkspaceStore, frame: FrameRecord, config: AutoLabelConfig) -> list[QCFinding]:
    return frame_findings(frame, config, store.load_aux("lidar", frame.frame_id) if frame.has_lidar else None)


@dataclass
class DatasetQC:
    frames: list[FrameRecord]
    approved: list[FrameRecord]
    findings: dict[str, list[QCFinding]]  # frame đã approve -> mọi finding (có cờ acked)
    tracks: list[dict]
    log: list[dict]
    audit: dict[str, dict]

    @property
    def open_by_frame(self) -> dict[str, list[QCFinding]]:
        return {fid: op for fid, fs in self.findings.items() if (op := open_findings(fs))}


def track_findings(approved: list[FrameRecord]) -> list[dict]:
    """Cùng track_id (một vật) mà mang lớp khác nhau ở các frame đã approve."""
    classes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for f in approved:
        for o in f.objects:
            if o.track_id and o.review.status == "approved":
                classes[o.track_id][o.review.final_label or o.label].append(f"{f.frame_id}#{o.object_id}")
    return [
        {
            "code": "TRACK_CLASS_INCONSISTENT",
            "track_id": tid,
            "classes": {c: refs for c, refs in by_class.items()},
            "message": "Cùng một vật mang lớp " + " / ".join(f"{c} ({len(r)} frame)" for c, r in by_class.items()),
        }
        for tid, by_class in sorted(classes.items())
        if len(by_class) > 1
    ]


def _same_box(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return len(a) == len(b) and all(abs(x - y) <= BOX_TOL for x, y in zip(a, b, strict=True))


def log_findings(approved: list[FrameRecord], corrections: list[dict]) -> list[dict]:
    """Đối chiếu dòng correction log cuối cùng của mỗi object với trạng thái đang lưu."""
    last = {(e["frame_id"], e["object_id"]): e for e in corrections}
    out = []
    for f in approved:
        for o in f.objects:
            r = o.review
            if r.status == "pending" or r.action in MACHINE_ACTIONS:
                continue
            e = last.get((f.frame_id, o.object_id))
            ref = {"frame_id": f.frame_id, "object_id": o.object_id}
            if e is None:
                out.append({**ref, "code": "LOG_MISSING", "message": f"{r.action} không có dòng correction log nào"})
                continue
            if r.status == "deleted":
                ok = e["human_action"] == "DELETE"
            else:
                ok = (
                    e["human_action"] == r.action
                    and e.get("final_class") == r.final_label
                    and _same_box(e.get("final_bbox"), r.final_bbox)
                )
            if not ok:
                logged = f"{e['human_action']} → {e.get('final_class')} {e.get('final_bbox')}"
                now = f"{r.action} → {r.final_label} {r.final_bbox}" if r.status != "deleted" else "DELETE"
                out.append({**ref, "code": "LOG_MISMATCH", "message": f"Log ghi {logged}, nhãn đang lưu {now}"})
    return out


def collect(store: WorkspaceStore, config: AutoLabelConfig) -> DatasetQC:
    frames = store.list_frames()
    approved = [f for f in frames if f.status == "approved"]
    return DatasetQC(
        frames=frames,
        approved=approved,
        findings={f.frame_id: load_frame_findings(store, f, config) for f in approved},
        tracks=track_findings(approved),
        log=log_findings(approved, store.corrections()),
        audit=audit_summary(store, config, frames),
    )


def _audit_detail(a: dict) -> str:
    if not a["required"] or not a["population"]:
        return a["note"]
    rate = "—" if a["error_rate"] is None else f"{a['error_rate']:.1%}"
    return f"{a['n']} mẫu, {a['errors']} sai ({rate}, CI95 {a['ci95'][0]:.1%}–{a['ci95'][1]:.1%}). {a['note']}"


def dataset_report(store: WorkspaceStore, config: AutoLabelConfig, qc: DatasetQC | None = None) -> dict:
    qc = qc or collect(store, config)
    open_by_frame = qc.open_by_frame
    open_all = [f for fs in open_by_frame.values() for f in fs]
    n_acked = sum(f.acked for fs in qc.findings.values() for f in fs)
    status = Counter(f.status for f in qc.frames)
    a_obj, a_frame = qc.audit["object"], qc.audit["frame"]
    checks = [
        {
            "id": "approved_frames",
            "label": "Có frame đã approve",
            "ok": bool(qc.approved),
            "detail": f"{len(qc.approved)}/{len(qc.frames)} frame",
        },
        {
            "id": "final_labels",
            "label": "Nhãn cuối không còn lỗi QC chưa xử lý",
            "ok": not open_all,
            "detail": f"{len(open_all)} lỗi ở {len(open_by_frame)} frame (frame đó sẽ không được xuất)"
            if open_all
            else f"0 lỗi · {n_acked} cảnh báo người đã xác nhận",
        },
        {
            "id": "track_consistency",
            "label": "Mỗi track giữ một lớp qua các frame",
            "ok": not qc.tracks,
            "detail": f"{len(qc.tracks)} track đổi lớp" if qc.tracks else "OK",
        },
        {
            "id": "audit_trail",
            "label": "Correction log khớp nhãn cuối",
            "ok": not qc.log,
            "detail": f"{len(qc.log)} object lệch log" if qc.log else "OK",
        },
        {
            "id": "audit_objects",
            "label": "Audit ngẫu nhiên object duyệt theo lô",
            "ok": a_obj["passed"],
            "detail": _audit_detail(a_obj),
        },
        {
            "id": "audit_frames",
            "label": "Audit ngẫu nhiên frame (vật bị sót)",
            "ok": a_frame["passed"],
            "detail": _audit_detail(a_frame),
        },
    ]
    return {
        "status": "READY" if all(c["ok"] for c in checks) else "NOT_READY",
        "checks": checks,
        "frames": {"total": len(qc.frames), **{k: status.get(k, 0) for k in ("auto", "editing", "approved")}},
        "coverage": round(len(qc.approved) / len(qc.frames), 4) if qc.frames else None,
        "n_open_findings": len(open_all),
        "n_acked_findings": n_acked,
        "open_by_code": dict(Counter(f.code for f in open_all).most_common()),
        "frames_with_open_findings": sorted(open_by_frame),
        "open_findings": [f.model_dump() for f in open_all[:500]],
        "track_findings": qc.tracks,
        "log_findings": qc.log[:500],
        "audit": qc.audit,
    }


def render_qa_report(report: dict, export: dict) -> str:
    """qa_report.md đi kèm bản export: đọc được bằng mắt, mọi con số lấy từ manifest."""
    fr = report["frames"]
    lines = [
        f"# Báo cáo QA — export {export['export_id']}",
        "",
        f"- Trạng thái phát hành: **{report['status']}**",
        f"- Xuất {export['n_frames']} frame, {export['n_objects']} object; "
        f"workspace có {fr['total']} frame ({fr['approved']} đã approve, {fr['editing']} đang sửa, {fr['auto']} chưa mở)",
        f"- Bỏ qua vì còn lỗi QC: {len(export['frames_skipped_qc'])} frame",
        "",
        "## Checklist",
        "",
        "| | Tiêu chí | Chi tiết |",
        "| :-: | :-- | :-- |",
    ]
    lines += [f"| {'✅' if c['ok'] else '❌'} | {c['label']} | {c['detail']} |" for c in report["checks"]]
    lines += ["", "## Nguồn gốc nhãn đã xuất", "", "| Nguồn | #object |", "| :-- | --: |"]
    lines += [f"| {k} | {v} |" for k, v in export["provenance"]["source"].items()]
    lines += ["", "| Hành động của người | #object |", "| :-- | --: |"]
    lines += [f"| {k} | {v} |" for k, v in export["provenance"]["human_action"].items()]
    lines += [
        "",
        "`BATCH_APPROVE` = duyệt theo lô, không ai xem riêng; độ tin của nhóm này đến từ audit ngẫu nhiên bên dưới.",
        "",
        "## Audit ngẫu nhiên",
        "",
        "| Loại | Tổng thể | Mẫu có kết quả | Sai | Tỉ lệ | CI95 | Đạt |",
        "| :-- | --: | --: | --: | --: | :-- | :-: |",
    ]
    for kind, name in (("object", "Object duyệt theo lô"), ("frame", "Frame (vật bị sót)")):
        a = report["audit"][kind]
        rate = "—" if a["error_rate"] is None else f"{a['error_rate']:.1%}"
        ci = f"{a['ci95'][0]:.1%}–{a['ci95'][1]:.1%}" if a["n"] else "—"
        lines.append(
            f"| {name} | {a['population']} | {a['n']} | {a['errors']} | {rate} | {ci} | {'✅' if a['passed'] else '❌'} |"
        )
    if report["open_by_code"]:
        lines += ["", "## Lỗi QC chưa xử lý (frame bị bỏ qua)", "", "| Code | # |", "| :-- | --: |"]
        lines += [f"| `{c}` | {n} |" for c, n in report["open_by_code"].items()]
    if report["track_findings"]:
        lines += ["", "## Track đổi lớp", ""]
        lines += [f"- `{t['track_id']}`: {t['message']}" for t in report["track_findings"][:50]]
    if report["log_findings"]:
        lines += ["", "## Lệch correction log", ""]
        lines += [
            f"- {e['frame_id']} #{e['object_id']} `{e['code']}`: {e['message']}" for e in report["log_findings"][:50]
        ]
    lines += ["", f"Tái lập: config sha256 `{export['config_sha256']}`, commit `{export.get('git_commit') or '—'}`."]
    return "\n".join(lines) + "\n"
