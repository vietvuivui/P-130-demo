"""QC nhãn cuối: kiểm lại nhãn SAU khi người sửa (khi approve frame, trước khi xuất) và nhãn của file kiểm nhanh.

QA Agent chấm rủi ro cho box detector lúc pipeline chạy. Sau đó người sửa box, đổi lớp, vẽ box mới, nhận box
nội suy — các nhãn cuối này chưa từng đi qua kiểm tra nào. Ở đây:

- Cấu trúc (error, phải sửa): lớp ngoài taxonomy, box hỏng, box ra ngoài ảnh.
- Kiểm lại (warning): chạy lại check hình học + LiDAR của QA Agent trên nhãn cuối, chỉ với nhãn khác cái QA Agent
  đã chấm (box người vẽ, box đã sửa, lớp đã đổi). Nhãn giữ nguyên thì issue của nó người đã thấy lúc duyệt.
- Mức frame (warning): hai box cùng lớp trùng nhau, hai box khác lớp chồng khít lên nhau.

Warning được xác nhận "đã kiểm, giữ nguyên" (ack) theo fingerprint trạng thái cuối của object: sửa box sau khi ack
thì fingerprint đổi và finding hiện lại.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from src.agents.nodes.issues import check_geometry
from src.agents.nodes.lidar import check_lidar
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, LabelObject, QCAck, QCFinding, QCSeverity
from src.services.geometry import clip_box, iou, iou_matrix

# Nhãn cuối coi là "chưa đổi" so với box QA Agent đã chấm khi IoU >= ngưỡng này
SAME_BOX_IOU = 0.99
# Box vượt biên ảnh quá ngần này (px) -> BOX_OUT_OF_IMAGE
OUT_OF_IMAGE_TOL = 1.0
# Người xem riêng từng object (khác duyệt theo lô / máy tự làm): được ưu tiên giữ khi hai box trùng nhau
INDIVIDUAL_ACTIONS = {"KEEP", "EDIT_BOX", "CHANGE_CLASS", "ADD_BOX"}


class QCError(ValueError):
    pass


@dataclass
class FinalLabel:
    """Nhãn cuối của một object: lớp/box sau khi người sửa, hoặc lớp/box gốc nếu chưa sửa."""

    object_id: str
    label: str
    bbox: list[float]
    score: float = 1.0
    # True: nhãn cuối khác cái QA Agent đã chấm (hoặc chưa từng được chấm) -> kiểm lại hình học + LiDAR
    changed: bool = True
    # Khi hai box trùng: giữ box người vẽ (3) > box người xem riêng (2) > box duyệt theo lô / đang chờ (1)
    priority: int = 1
    # Mã lỗi người đã xem và quyết định giữ nguyên nhãn này (file kiểm nhanh ghi lại): không báo lại
    seen: frozenset[str] = frozenset()


@dataclass
class FrameContext:
    frame_id: str
    width: int | None = None
    height: int | None = None
    fy: float | None = None
    uv: np.ndarray | None = None
    depth: np.ndarray | None = None

    @classmethod
    def from_frame(cls, frame: FrameRecord, lidar_aux: dict | None) -> FrameContext:
        uv, depth = lidar_arrays(lidar_aux) if frame.has_lidar else (None, None)
        return cls(frame.frame_id, frame.image.width, frame.image.height, frame.intrinsic[1][1], uv, depth)


def lidar_arrays(aux: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Điểm LiDAR đã chiếu xuống ảnh mà pipeline lưu ở workspace/lidar/<frame>.json."""
    if not aux or not aux.get("u"):
        return None, None
    uv = np.column_stack([aux["u"], aux["v"]]).astype(np.float64)
    return uv, np.asarray(aux["d"], dtype=np.float64)


def final_labels(frame: FrameRecord) -> list[FinalLabel]:
    out = []
    for o in frame.objects:
        if o.review.status == "deleted":
            continue
        label = o.review.final_label or o.label
        bbox = list(o.review.final_bbox or o.bbox)
        changed = o.source == "human" or o.qa is None or label != o.label or iou(bbox, o.bbox) < SAME_BOX_IOU
        priority = 3 if o.source == "human" else 2 if o.review.action in INDIVIDUAL_ACTIONS else 1
        out.append(FinalLabel(o.object_id, label, bbox, o.score, changed, priority))
    return out


def fingerprint(labels: list[FinalLabel]) -> str:
    data = sorted((lab.object_id, lab.label, [round(v, 1) for v in lab.bbox]) for lab in labels)
    return hashlib.sha1(json.dumps(data).encode()).hexdigest()[:12]


def _finding(
    code: str,
    severity: QCSeverity,
    message: str,
    frame_id: str,
    labels: list[FinalLabel],
    suggestion: dict | None = None,
) -> QCFinding:
    """Finding trên 1 object (labels[0]) hoặc một cặp (labels[0] là object cần xử lý, labels[1] là object còn lại)."""
    ids = [lab.object_id for lab in labels]
    return QCFinding(
        code=code,
        severity=severity,
        message=message,
        frame_id=frame_id,
        object_id=ids[0],
        other_object_id=ids[1] if len(ids) > 1 else None,
        suggestion=suggestion,
        # Cặp: key không phụ thuộc thứ tự để ack không mất khi độ ưu tiên hai box đổi
        key=":".join([code, *sorted(ids)]) if len(ids) > 1 else f"{code}:{ids[0]}",
        fingerprint=fingerprint(labels),
    )


def _structural(lab: FinalLabel, ctx: FrameContext, config: AutoLabelConfig) -> list[QCFinding]:
    out = []
    if lab.label not in config.classes:
        out.append(
            _finding("UNKNOWN_CLASS", "error", f"Lớp '{lab.label}' không có trong taxonomy", ctx.frame_id, [lab])
        )
    b = lab.bbox
    if len(b) != 4 or not all(isinstance(v, int | float) and math.isfinite(v) for v in b):
        out.append(_finding("INVALID_BOX", "error", "Box phải là 4 số hữu hạn [x1, y1, x2, y2]", ctx.frame_id, [lab]))
        return out
    min_px = config.qc.min_box_px
    w, h = b[2] - b[0], b[3] - b[1]
    if w < min_px or h < min_px:
        why = "toạ độ ngược (x2 < x1 hoặc y2 < y1)" if w < 0 or h < 0 else f"cạnh phải ≥ {min_px:g}px"
        out.append(_finding("INVALID_BOX", "error", f"Box {w:.1f}×{h:.1f}px: {why}", ctx.frame_id, [lab]))
        return out
    if ctx.width and ctx.height:
        clipped = [round(v, 1) for v in clip_box(b, ctx.width, ctx.height)]
        if clipped[2] - clipped[0] < min_px or clipped[3] - clipped[1] < min_px:
            out.append(_finding("INVALID_BOX", "error", "Box nằm ngoài ảnh", ctx.frame_id, [lab]))
        elif any(abs(v - c) > OUT_OF_IMAGE_TOL for v, c in zip(b, clipped, strict=True)):
            out.append(
                _finding(
                    "BOX_OUT_OF_IMAGE",
                    "error",
                    f"Box vượt ra ngoài ảnh {ctx.width}×{ctx.height}",
                    ctx.frame_id,
                    [lab],
                    {"action": "EDIT_BOX", "object_id": lab.object_id, "bbox": clipped},
                )
            )
    return out


def _recheck(lab: FinalLabel, ctx: FrameContext, config: AutoLabelConfig) -> list[QCFinding]:
    """Chạy lại check hình học + LiDAR của QA Agent trên nhãn cuối."""
    box = clip_box(lab.bbox, ctx.width, ctx.height) if ctx.width and ctx.height else lab.bbox
    issues = []
    if ctx.width and ctx.height:
        obj = LabelObject(object_id=lab.object_id, bbox=box, label=lab.label, score=lab.score)
        issues += check_geometry(obj, ctx.width, ctx.height, config)["issues"]
    if ctx.uv is not None and ctx.fy and ctx.height:
        issues += check_lidar(box, lab.label, ctx.uv, ctx.depth, ctx.fy, ctx.height, config)["issues"]
    return [_finding(i.code, "warning", f"Nhãn cuối: {i.message}", ctx.frame_id, [lab]) for i in issues]


def _overlaps(labels: list[FinalLabel], frame_id: str, config: AutoLabelConfig) -> list[QCFinding]:
    """Box trùng cùng lớp (một vật gán hai lần) và box khác lớp chồng khít (một vật gán hai lớp)."""
    if len(labels) < 2:
        return []
    cfg = config.qc
    allowed = {frozenset(p) for p in cfg.cross_class_allowed}
    m = iou_matrix(np.array([lab.bbox for lab in labels]), np.array([lab.bbox for lab in labels]))
    out = []
    for i, j in zip(*np.triu_indices(len(labels), k=1), strict=True):
        a, b, v = labels[i], labels[j], float(m[i, j])
        weak, strong = (a, b) if (a.priority, a.score) < (b.priority, b.score) else (b, a)
        if a.label == b.label and v >= cfg.duplicate_iou:
            out.append(
                _finding(
                    "DUPLICATE_BOX",
                    "warning",
                    f"Trùng #{strong.object_id} ({strong.label}), IoU {v:.2f}: một vật bị gán hai lần",
                    frame_id,
                    [weak, strong],
                    {"action": "DELETE", "object_id": weak.object_id},
                )
            )
        elif a.label != b.label and v >= cfg.cross_class_iou and frozenset((a.label, b.label)) not in allowed:
            out.append(
                _finding(
                    "OVERLAP_CROSS_CLASS",
                    "warning",
                    f"Chồng khít #{strong.object_id} ({strong.label}), IoU {v:.2f}: một vật bị gán hai lớp "
                    f"({weak.label} / {strong.label})",
                    frame_id,
                    [weak, strong],
                )
            )
    return out


def check_labels(labels: list[FinalLabel], ctx: FrameContext, config: AutoLabelConfig) -> list[QCFinding]:
    findings, valid = [], []
    for lab in labels:
        structural = _structural(lab, ctx, config)
        findings += structural
        if any(f.code == "INVALID_BOX" for f in structural):
            continue
        valid.append(lab)
        if lab.changed and lab.label in config.classes:
            findings += _recheck(lab, ctx, config)
    return findings + _overlaps(valid, ctx.frame_id, config)


def apply_acks(findings: list[QCFinding], acks: list[QCAck]) -> list[QCFinding]:
    """Đánh dấu warning người đã xác nhận. Error không ack được; ack chỉ khớp khi nhãn chưa đổi (fingerprint)."""
    done = {(a.key, a.fingerprint) for a in acks}
    return [
        f.model_copy(update={"acked": f.severity == "warning" and (f.key, f.fingerprint) in done}) for f in findings
    ]


def frame_findings(frame: FrameRecord, config: AutoLabelConfig, lidar_aux: dict | None) -> list[QCFinding]:
    """Mọi finding QC của nhãn cuối trong frame (kể cả finding audit gắn tay), đã đánh dấu acked."""
    found = check_labels(final_labels(frame), FrameContext.from_frame(frame, lidar_aux), config)
    return apply_acks(found + [m.model_copy() for m in frame.qc_manual], frame.qc_acks)


def open_findings(findings: list[QCFinding]) -> list[QCFinding]:
    return [f for f in findings if not f.acked]


def ack_finding(
    frame: FrameRecord, findings: list[QCFinding], key: str, fp: str, reviewer: str, note: str, at: str
) -> QCFinding:
    """Người xác nhận "đã kiểm, giữ nguyên" một warning. Ghi vào frame; route ghi thêm nhật ký QC."""
    match = next((f for f in findings if f.key == key and f.fingerprint == fp), None)
    if match is None:
        raise QCError("Finding không còn (nhãn đã đổi sau khi tải QC). Tải lại QC của frame rồi xác nhận lại")
    if match.severity == "error":
        raise QCError(f"{match.code} là lỗi phải sửa, không xác nhận bỏ qua được")
    frame.qc_acks = [a for a in frame.qc_acks if a.key != key] + [
        QCAck(key=key, fingerprint=fp, reviewer=reviewer, at=at, note=note)
    ]
    return match
