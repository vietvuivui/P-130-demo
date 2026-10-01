"""Sửa nhãn ở các sweep lân cận keyframe (t-2 … t+2) và tính lại QA / score của keyframe.

Sweep không được xuất ra (nuScenes chỉ gán nhãn keyframe 2Hz); box ở sweep là bằng chứng cho keyframe: FLICKER,
RECOVERED_BY_TRACK, điểm temporal trong risk, score tính lại theo sweep (temporal_fusion.py) và tracker lan truyền.
Detector sót hoặc nhận sai ở sweep làm các bằng chứng đó sai theo, nên người được sửa tự do như ở keyframe: giữ, xoá,
đổi lớp, sửa box, vẽ thêm. Sửa xong, keyframe được tính lại ngay; quyết định của người ở keyframe giữ nguyên.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.agents.nodes.confidence import check_confidence
from src.agents.nodes.issues import check_geometry
from src.agents.nodes.lidar import check_lidar
from src.agents.nodes.risk import score_risk
from src.agents.nodes.temporal import check_temporal
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import (
    Detection,
    FrameRecord,
    LabelObject,
    QAResult,
    ReviewState,
    SweepActionRequest,
    SweepBox,
    SweepInfo,
)
from src.services.geometry import clip_box, iou
from src.services.review import ReviewError, _mark_editing, now_iso
from src.services.temporal_fusion import rescore


def effective_detections(sweep: SweepInfo) -> list[Detection]:
    """Box của sweep sau khi người sửa (hoặc detection máy nếu chưa sửa). Box người đã xác nhận có score 1."""
    if sweep.boxes is None:
        return list(sweep.detections)
    out = []
    for b in sweep.boxes:
        if b.review.status == "deleted":
            continue
        confirmed = b.review.status == "approved" or b.source == "human"
        out.append(
            Detection(
                bbox=b.review.final_bbox or b.bbox,
                label=b.review.final_label or b.label,
                score=1.0 if confirmed else b.score,
            )
        )
    return out


def ensure_boxes(sweep: SweepInfo) -> list[SweepBox]:
    if sweep.boxes is None:
        sweep.boxes = [
            SweepBox(box_id=str(i + 1), bbox=d.bbox, label=d.label, score=d.score)
            for i, d in enumerate(sweep.detections)
        ]
    return sweep.boxes


def find_sweep(frame: FrameRecord, offset: int) -> SweepInfo:
    for s in frame.sweeps:
        if s.offset == offset:
            return s
    raise ReviewError(f"Frame {frame.frame_id} không có sweep t{offset:+d}")


def apply_sweep_action(
    frame: FrameRecord, offset: int, req: SweepActionRequest, reviewer: str, config: AutoLabelConfig
) -> dict:
    """Áp một thao tác lên một box của sweep (sửa tại chỗ). Trả về dòng log."""
    _mark_editing(frame)
    sweep = find_sweep(frame, offset)
    boxes = ensure_boxes(sweep)
    if req.label is not None and req.label not in config.classes:
        raise ReviewError(f"Lớp không có trong taxonomy: {req.label}")
    bbox = [round(v, 1) for v in clip_box(req.bbox, sweep.width, sweep.height)] if req.bbox else None
    if bbox and (bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2):
        raise ReviewError("Box quá nhỏ hoặc nằm ngoài ảnh")

    if req.action == "ADD_BOX":
        if bbox is None or req.label is None:
            raise ReviewError("ADD_BOX cần bbox và label")
        n = sum(b.source == "human" for b in boxes) + 1
        box = SweepBox(box_id=f"h{n}", bbox=bbox, label=req.label, score=1.0, source="human")
        boxes.append(box)
    else:
        box = next((b for b in boxes if b.box_id == req.box_id), None)
        if box is None:
            raise ReviewError(f"Sweep t{offset:+d} không có box {req.box_id}")
    cur_label, cur_bbox = box.review.final_label or box.label, box.review.final_bbox or box.bbox
    if req.action == "DELETE":
        box.review = ReviewState(status="deleted", action="DELETE")
    elif req.action == "RESTORE":
        box.review = ReviewState()
    elif req.action == "KEEP":
        box.review = ReviewState(status="approved", action="KEEP", final_label=cur_label, final_bbox=cur_bbox)
    elif req.action == "CHANGE_CLASS":
        if req.label is None:
            raise ReviewError("CHANGE_CLASS cần label")
        box.review = ReviewState(status="approved", action="CHANGE_CLASS", final_label=req.label, final_bbox=cur_bbox)
    elif req.action == "EDIT_BOX":
        if bbox is None:
            raise ReviewError("EDIT_BOX cần bbox")
        box.review = ReviewState(
            status="approved", action="EDIT_BOX", final_label=req.label or cur_label, final_bbox=bbox
        )
    elif req.action == "ADD_BOX":
        box.review = ReviewState(status="approved", action="ADD_BOX", final_label=req.label, final_bbox=bbox)
    box.review.reviewer, box.review.at = reviewer, now_iso()
    return {
        "frame_id": frame.frame_id,
        "sweep_offset": offset,
        "box_id": box.box_id,
        "prediction": {"bbox": box.bbox, "class": box.label, "score": box.score, "source": box.source},
        "human_action": req.action,
        "final_class": box.review.final_label,
        "final_bbox": box.review.final_bbox,
        "reviewer": reviewer,
        "timestamp": box.review.at,
    }


def _qa(conf: dict, lid: dict, tmp: dict, geo: dict) -> QAResult:
    return QAResult(
        risk=0.0,
        level="low",
        issues=conf["issues"] + lid["issues"] + tmp["issues"] + geo["issues"],
        terms={
            "detection": round(conf["term"], 3),
            "lidar": round(lid["term"], 3),
            "temporal": round(tmp["term"], 3),
            "geometric": round(geo["term"], 3),
        },
        lidar={k: v for k, v in lid.items() if k not in ("issues", "term")},
        temporal={k: v for k, v in tmp.items() if k not in ("issues", "term", "track")},
    )


def _kept_lidar(obj: LabelObject, fallback: Callable[[], dict]) -> dict:
    """Kết quả check LiDAR lúc auto-label (point cloud đầy đủ); chỉ tính lại từ bản rút gọn nếu chưa có."""
    if obj.qa is None:
        return fallback()
    return {
        **obj.qa.lidar,
        "issues": [i for i in obj.qa.issues if i.group == "lidar"],
        "term": obj.qa.terms.get("lidar", 0.0),
    }


def requalify_frame(
    frame: FrameRecord,
    config: AutoLabelConfig,
    image_file: Callable[[str], Path],
    uv=None,
    depth=None,
) -> None:
    """Tính lại score (theo sweep), check temporal, đề xuất RECOVERED_BY_TRACK và risk của keyframe từ box sweep hiện
    tại (sau khi người sửa). Object máy sinh giữ id và quyết định của người; đề xuất nội suy cũ còn chờ được thay mới."""
    from src.services.pipeline import sweep_state, sweep_warps

    tcfg = config.qa.temporal
    sweeps = {s.offset: s for s in frame.sweeps}
    sweep_dets = {o: effective_detections(s) for o, s in sweeps.items()}
    warped = sweep_warps(frame.image, sweeps, sweep_dets, image_file, tcfg) if sweeps else {}

    models = [o for o in frame.objects if o.source == "model"]
    base = [
        Detection(bbox=o.bbox, label=o.label, score=o.det_score if o.det_score is not None else o.score) for o in models
    ]
    scored = rescore(base, sweep_dets, warped, tcfg.rescore, tcfg.match_iou)
    for o, d in zip(models, scored, strict=True):
        o.score, o.det_score = d.score, d.det_score

    state = {o: sweep_state(sweeps[o].timestamp, sweep_dets[o], warped.get(o)) for o in sweeps}
    results, recovered = check_temporal(models, state, frame.image.timestamp, config)
    fy, (w, h) = frame.intrinsic[1][1], (frame.image.width, frame.image.height)

    def lidar_of(obj):
        return check_lidar(obj.bbox, obj.label, uv, depth, fy, h, config)

    for o in models:
        tmp = results[o.object_id]
        qa = _qa(
            check_confidence(o, config), _kept_lidar(o, lambda o=o: lidar_of(o)), tmp, check_geometry(o, w, h, config)
        )
        o.qa = score_risk(qa, config.qa.risk)
        o.track = tmp.get("track", o.track)

    # Đề xuất nội suy: giữ cái người đã xử lý, thay cái còn chờ bằng đề xuất mới
    keep = [o for o in frame.objects if not (o.source == "track" and o.review.status == "pending")]
    taken = [o.review.final_bbox or o.bbox for o in keep if o.review.status != "deleted" and o.source != "model"]
    used_ids = {o.object_id for o in keep}
    n = 0
    for r in recovered:
        if any(iou(r.bbox, b) >= tcfg.match_iou for b in taken):
            continue
        tmp = results[r.object_id]
        n += 1
        while f"r{n}" in used_ids:
            n += 1
        qa = _qa(check_confidence(r, config), lidar_of(r), tmp, check_geometry(r, w, h, config))
        keep.append(r.model_copy(update={"object_id": f"r{n}", "qa": score_risk(qa, config.qa.risk)}))
        used_ids.add(f"r{n}")
        taken.append(r.bbox)
    frame.objects = keep
    frame.frame_risk = max((o.qa.risk for o in frame.objects if o.qa and o.source in ("model", "track")), default=0.0)
