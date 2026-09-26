"""3.4 Rule-based issue generation: gộp kết quả 3 check + kiểm tra hình học."""

from __future__ import annotations

from src.agents.nodes.confidence import check_confidence
from src.agents.nodes.lidar import check_lidar
from src.agents.state import QAState
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import LabelObject, QAIssue, QAResult
from src.services.geometry import box_area, touches_border


def check_geometry(obj: LabelObject, width: int, height: int, config: AutoLabelConfig) -> dict:
    cfg = config.qa.geometry
    issues, term = [], 0.0
    x1, y1, x2, y2 = obj.bbox

    area_ratio = box_area(obj.bbox) / float(width * height)
    if area_ratio > cfg.max_area_ratio:
        issues.append(
            QAIssue(code="BOX_TOO_LARGE", group="geometric", message=f"Box chiếm {area_ratio:.0%} diện tích ảnh")
        )
        term = 1.0

    spec = config.classes.get(obj.label)
    h = y2 - y1
    # Object bị cắt ở mép ảnh vẫn là nhãn hợp lệ nhưng tỉ lệ bị méo, nên bỏ qua
    if spec and h > 0 and not touches_border(obj.bbox, width, height, cfg.border_margin_px):
        ar = (x2 - x1) / h
        lo, hi = spec.aspect_wh
        if not lo <= ar <= hi:
            issues.append(
                QAIssue(
                    code="ASPECT_RATIO_ABNORMAL",
                    group="geometric",
                    message=f"Tỉ lệ rộng/cao {ar:.2f} ngoài [{lo}, {hi}] của lớp {obj.label}",
                )
            )
            term = max(term, 0.7)
    return {"issues": issues, "term": term}


def issue_node(state: QAState) -> dict:
    config = state["config"]
    width, height = state["image_size"]
    fy = state["intrinsic"][1][1]
    uv, depth = state.get("lidar_uv"), state.get("lidar_depth")
    confidence, lidar, temporal = state["confidence"], state["lidar"], state["temporal"]

    reviewed = []
    for obj in list(state.get("objects", [])) + list(state.get("recovered", [])):
        oid = obj.object_id
        # Box đề xuất từ tracking chưa đi qua check song song, chạy bù ở đây
        conf = confidence.get(oid) or check_confidence(obj, config)
        lid = lidar.get(oid) or check_lidar(obj.bbox, obj.label, uv, depth, fy, height, config)
        tmp = temporal.get(oid, {"issues": [], "term": 0.0})
        geo = check_geometry(obj, width, height, config)

        qa = QAResult(
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
        track = obj.track or tmp.get("track", {})
        reviewed.append(obj.model_copy(update={"qa": qa, "track": track}))
    return {"reviewed": reviewed}
