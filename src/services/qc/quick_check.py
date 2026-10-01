"""Quick Check: kiểm nhanh một file nhãn từ ngoài, không ghi gì vào workspace.

Dùng khi nhận nhãn của người gán thuê ngoài / tool khác, hoặc để kiểm lại một bản export cũ theo config hiện tại
(export -> Quick Check file coco.json phải ra 0 lỗi). Không cần tải ảnh lên: chỉ file nhãn.

Định dạng nhận:
- COCO (.json có images + annotations): bbox [x, y, w, h]; ảnh khớp workspace theo `frame_id` (export của repo này
  có sẵn) hoặc theo tên file ảnh.
- JSONL (.jsonl, như labels.jsonl của export): mỗi dòng {"frame_id", "objects": [{"bbox": [x1, y1, x2, y2], "class"}]}.
- JSON {"frames": [...]} hoặc danh sách frame, cùng dạng với một dòng JSONL.

Frame có trong workspace thì kiểm thêm bằng dữ liệu đã lưu: LiDAR (NO_LIDAR_SUPPORT, SIZE_DEPTH_MISMATCH), detection
của model (POSSIBLY_MISSING: vật detector thấy ổn định mà file không có; MODEL_DISAGREES: lớp khác model).

Mọi nhãn trong file đều được kiểm lại (không biết ai tạo ra). Riêng cảnh báo mà file ghi là người đã xem và giữ nguyên
(`qc_acknowledged`, hoặc `issues` của nhãn `human_action: KEEP` — export của repo này ghi đủ) thì đánh dấu acked.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, QCFinding, QuickCheckFrame, QuickCheckLabel, QuickCheckResponse
from src.services.evaluation import prelabel_objects
from src.services.geometry import iou, iou_matrix
from src.services.qc.checks import FinalLabel, FrameContext, QCError, check_labels, fingerprint
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore

# Nhãn khớp detection ở IoU này mới so lớp (MODEL_DISAGREES); detection trùng box người đã xoá thì không gợi ý thêm
MATCH_IOU = 0.5


@dataclass
class ExternalFrame:
    frame_id: str | None = None
    file_name: str | None = None
    width: int | None = None
    height: int | None = None
    labels: list[FinalLabel] = field(default_factory=list)
    # (object_id, lý do) của nhãn không đọc được
    bad: list[tuple[str, str]] = field(default_factory=list)


def _class_name(raw, config: AutoLabelConfig) -> str:
    """Tên lớp ngoài -> lớp nội bộ: tên lớp, category nuScenes (vehicle.car) hoặc prompt (person, van...)."""
    s = str(raw).strip()
    if s in config.classes:
        return s
    if s.lower() in config.classes:
        return s.lower()
    if s in config.gt_category_map:
        return config.gt_category_map[s]
    return config.prompt_to_class().get(s.lower(), s)


def _numbers(v) -> list[float] | None:
    if not isinstance(v, list | tuple) or len(v) != 4:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _seen(o: dict) -> frozenset[str]:
    """Mã lỗi file ghi là người đã xem mà vẫn giữ nhãn (export của repo này ghi đủ): không báo lại.

    - qc_acknowledged: cảnh báo QC người đã xác nhận "đã kiểm, giữ nguyên".
    - issues + human_action KEEP: người thấy issue của QA Agent mà vẫn giữ nguyên box và lớp. Sửa box / đổi lớp
      thì issue cũ nói về nhãn cũ, không dùng được.
    """
    codes = o.get("qc_acknowledged") or []
    if o.get("human_action") == "KEEP":
        codes = [*codes, *(o.get("issues") or [])]
    return frozenset(str(c) for c in codes) if isinstance(codes, list) else frozenset()


def _frame_entry(row: dict, n: int, config: AutoLabelConfig) -> ExternalFrame:
    """Một frame dạng {"frame_id", "image"?, "width"?, "height"?, "objects": [{"bbox": xyxy, "class"}]}."""
    ext = ExternalFrame(
        frame_id=row.get("frame_id"),
        file_name=row.get("image") or row.get("file_name"),
        width=_int_or_none(row.get("width")),
        height=_int_or_none(row.get("height")),
    )
    objects = row.get("objects", row.get("labels", []))
    if not isinstance(objects, list):
        ext.bad.append(("-", f"frame #{n}: 'objects' phải là danh sách"))
        return ext
    for i, o in enumerate(objects, start=1):
        oid = str(o.get("object_id", i)) if isinstance(o, dict) else str(i)
        box = _numbers(o.get("bbox")) if isinstance(o, dict) else None
        cls = (o.get("class") or o.get("label")) if isinstance(o, dict) else None
        if box is None or cls is None:
            ext.bad.append((oid, "cần 'bbox' [x1, y1, x2, y2] (4 số) và 'class'"))
            continue
        ext.labels.append(FinalLabel(oid, _class_name(cls, config), box, seen=_seen(o)))
    return ext


def _parse_coco(doc: dict, config: AutoLabelConfig) -> list[ExternalFrame]:
    cats = {c.get("id"): c.get("name") for c in doc.get("categories", []) if isinstance(c, dict)}
    frames: dict = {}
    for img in doc.get("images", []):
        if not isinstance(img, dict):
            continue
        frames[img.get("id")] = ExternalFrame(
            frame_id=img.get("frame_id"),
            file_name=img.get("file_name"),
            width=_int_or_none(img.get("width")),
            height=_int_or_none(img.get("height")),
        )
    orphan = ExternalFrame(frame_id=None, file_name="(annotation không có image_id hợp lệ)")
    for n, ann in enumerate(doc.get("annotations", []), start=1):
        if not isinstance(ann, dict):
            continue
        ext = frames.get(ann.get("image_id"), orphan)
        oid = str(ann.get("object_id") or ann.get("id") or n)
        xywh = _numbers(ann.get("bbox"))
        cls = cats.get(ann.get("category_id"))
        if xywh is None or cls is None:
            ext.bad.append((oid, "cần 'bbox' [x, y, w, h] và 'category_id' có trong categories"))
            continue
        x, y, w, h = xywh
        ext.labels.append(FinalLabel(oid, _class_name(cls, config), [x, y, x + w, y + h], seen=_seen(ann)))
    out = list(frames.values())
    return out + ([orphan] if orphan.bad or orphan.labels else [])


def parse_labels(name: str, data: bytes, config: AutoLabelConfig) -> tuple[str, list[ExternalFrame], list[str]]:
    """Đọc file nhãn -> (định dạng, danh sách frame, lỗi đọc không gắn được vào frame nào)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise QCError("File phải là văn bản UTF-8 (JSON / JSONL)") from e

    errors: list[str] = []
    if name.lower().endswith(".jsonl"):
        rows = []
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Dòng {n}: JSON hỏng ({e.msg})")
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append(f"Dòng {n}: phải là một object JSON")
        return "jsonl", [_frame_entry(r, n, config) for n, r in enumerate(rows, start=1)], errors

    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise QCError(f"JSON hỏng ở dòng {e.lineno}: {e.msg}") from e
    if isinstance(doc, dict) and "annotations" in doc and "images" in doc:
        return "coco", _parse_coco(doc, config), errors
    rows = doc.get("frames") if isinstance(doc, dict) else doc
    if not isinstance(rows, list):
        raise QCError(
            "Không nhận ra định dạng: cần COCO (images + annotations), {'frames': [...]} hoặc danh sách frame"
        )
    frames = [_frame_entry(r, n, config) for n, r in enumerate(rows, start=1) if isinstance(r, dict)]
    return "frames", frames, errors


def _model_findings(labels: list[FinalLabel], ws: FrameRecord, config: AutoLabelConfig) -> list[QCFinding]:
    """So file nhãn với detection đã lưu của frame: vật bị sót và lớp khác model."""
    qc = config.qc.quick_check
    min_support = config.qa.temporal.min_support
    dets = [o for o in prelabel_objects(ws) if o.source == "model"]
    # Box người đã xoá trong workspace (FP đã xác nhận) thì không gợi ý thêm lại
    rejected = [o.bbox for o in ws.objects if o.review.action == "DELETE"]
    if not dets:
        return []
    boxes = [lab.bbox for lab in labels]
    m = iou_matrix(np.array([d.bbox for d in dets]), np.array(boxes)) if boxes else np.zeros((len(dets), 0))
    out: dict[str, QCFinding] = {}
    for i, d in enumerate(dets):
        det = FinalLabel(f"det{d.object_id}", d.label, list(d.bbox), d.score)
        best_j = int(m[i].argmax()) if boxes else -1
        best = float(m[i, best_j]) if boxes else 0.0
        if best < qc.missing_match_iou:
            tmp = d.qa.temporal if d.qa else {}
            support, available = tmp.get("support"), tmp.get("available")
            stable = support is None or not available or support >= min_support
            if d.score < qc.missing_min_score or not stable or any(iou(d.bbox, r) >= MATCH_IOU for r in rejected):
                continue
            seen = f", có ở {support}/{available} sweep lân cận" if available else ""
            out[f"POSSIBLY_MISSING:{det.object_id}"] = QCFinding(
                code="POSSIBLY_MISSING",
                severity="warning",
                message=f"Detector thấy {d.label} (score {d.score:.2f}{seen}) nhưng file không có box nào ở đây",
                frame_id=ws.frame_id,
                suggestion={"action": "ADD_BOX", "bbox": [round(v, 1) for v in d.bbox], "label": d.label},
                key=f"POSSIBLY_MISSING:{det.object_id}",
                fingerprint=fingerprint([det]),
            )
            continue
        lab = labels[best_j]
        if best >= MATCH_IOU and d.label != lab.label and d.score >= qc.disagree_min_score:
            key = f"MODEL_DISAGREES:{lab.object_id}"
            prev = out.get(key)
            if prev is None or d.score > prev.suggestion["score"]:
                out[key] = QCFinding(
                    code="MODEL_DISAGREES",
                    severity="warning",
                    message=f"File gán {lab.label}, detector gán {d.label} (score {d.score:.2f}, IoU {best:.2f})",
                    frame_id=ws.frame_id,
                    object_id=lab.object_id,
                    suggestion={
                        "action": "CHANGE_CLASS",
                        "object_id": lab.object_id,
                        "label": d.label,
                        "score": round(d.score, 3),
                    },
                    key=key,
                    fingerprint=fingerprint([lab]),
                )
    return list(out.values())


def _workspace_index(store: WorkspaceStore) -> tuple[dict[str, FrameRecord], dict[str, str]]:
    frames = {f.frame_id: f for f in store.list_frames()}
    by_name = {}
    for f in frames.values():
        path = f.image.path.removeprefix(WORKSPACE_PREFIX)
        by_name[path] = by_name[Path(path).name] = f.frame_id
    return frames, by_name


def check_frame(ext: ExternalFrame, ws: FrameRecord | None, lidar_aux: dict | None, config: AutoLabelConfig, n: int):
    if ws is not None:
        ctx = FrameContext.from_frame(ws, lidar_aux)
    else:
        ctx = FrameContext(ext.frame_id or ext.file_name or f"frame#{n}", ext.width, ext.height)
    findings = [
        QCFinding(
            code="INVALID_BOX",
            severity="error",
            message=f"Không đọc được nhãn: {why}",
            frame_id=ctx.frame_id,
            object_id=oid,
            key=f"INVALID_BOX:{oid}",
            fingerprint="parse",
        )
        for oid, why in ext.bad
    ]
    findings += check_labels(ext.labels, ctx, config)
    if ws is not None:
        findings += _model_findings(ext.labels, ws, config)
    else:
        what = "cấu trúc, box trùng" + (" và hình học" if ctx.width and ctx.height else "")
        findings.append(
            QCFinding(
                code="FRAME_NOT_IN_WORKSPACE",
                severity="warning",
                message=f"Frame không có trong workspace: chỉ kiểm {what} (không có LiDAR / detection để đối chiếu)",
                frame_id=ctx.frame_id,
                key="FRAME_NOT_IN_WORKSPACE",
                fingerprint="-",
            )
        )
    # Cảnh báo người đã xem và giữ nguyên (file ghi lại) -> đánh dấu acked, không tính là lỗi mở
    seen = {lab.object_id: lab.seen for lab in ext.labels}
    findings = [
        f.model_copy(update={"acked": True}) if f.severity == "warning" and f.code in seen.get(f.object_id, ()) else f
        for f in findings
    ]
    return QuickCheckFrame(
        frame_id=ws.frame_id if ws else ext.frame_id,
        file_name=ext.file_name,
        in_workspace=ws is not None,
        width=ctx.width,
        height=ctx.height,
        labels=[QuickCheckLabel(object_id=lab.object_id, label=lab.label, bbox=lab.bbox) for lab in ext.labels],
        findings=findings,
    )


def quick_check(store: WorkspaceStore, config: AutoLabelConfig, name: str, data: bytes) -> QuickCheckResponse:
    t0 = time.perf_counter()
    fmt, frames, parse_errors = parse_labels(name, data, config)
    ws_frames, by_name = _workspace_index(store)
    results = []
    for n, ext in enumerate(frames, start=1):
        fid = ext.frame_id if ext.frame_id in ws_frames else None
        if fid is None and ext.file_name:
            name_key = str(ext.file_name).removeprefix(WORKSPACE_PREFIX)
            fid = by_name.get(name_key) or by_name.get(Path(name_key).name)
        ws = ws_frames.get(fid) if fid else None
        lidar = store.load_aux("lidar", ws.frame_id) if ws is not None and ws.has_lidar else None
        results.append(check_frame(ext, ws, lidar, config, n))

    open_findings = [f for r in results for f in r.findings if not f.acked]
    return QuickCheckResponse(
        file_name=name,
        format=fmt,
        n_frames=len(results),
        n_labels=sum(len(r.labels) for r in results),
        n_errors=sum(f.severity == "error" for f in open_findings),
        n_warnings=sum(f.severity == "warning" for f in open_findings),
        n_acked=sum(f.acked for r in results for f in r.findings),
        by_code=dict(Counter(f.code for f in open_findings).most_common()),
        # Frame nhiều lỗi lên đầu
        frames=sorted(
            results,
            key=lambda r: (
                -sum(f.severity == "error" for f in r.findings),
                -sum(not f.acked for f in r.findings),
                r.frame_id or "",
            ),
        ),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        parse_errors=parse_errors,
    )
