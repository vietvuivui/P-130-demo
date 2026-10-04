"""3.1 Confidence check: score thấp và xung đột lớp."""

from __future__ import annotations

from src.agents.state import QAState
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import LabelObject, QAIssue


def check_confidence(obj: LabelObject, config: AutoLabelConfig) -> dict:
    cfg = config.qa.confidence
    issues = []
    # Box chỉ camera thấy trong dự án có LiDAR: score đã bị hạ khi gộp với box 3D (lidar2d.py), det_score là điểm gốc
    # của detector. Việc LiDAR không xác nhận đã được tính ở check LiDAR (số điểm trong box); nếu còn dùng score đã hạ để
    # xét LOW_CONFIDENCE thì phạt hai lần, và xe detector thấy rõ (0.9) cũng thành rủi ro cao.
    camera_only = (
        cfg.camera_only_uses_det_score
        and obj.box3d is None
        and obj.det_score is not None
        and obj.det_score > obj.score
    )
    score = obj.det_score if camera_only else obj.score
    if score < cfg.low_threshold:
        issues.append(
            QAIssue(code="LOW_CONFIDENCE", group="detection", message=f"Score {score:.2f} < {cfg.low_threshold:.2f}")
        )
    if camera_only:  # có issue nên không bao giờ vào nhóm duyệt theo lô (risk.issue_floor)
        issues.append(
            QAIssue(
                code="CAMERA_ONLY",
                group="lidar",
                message=f"Chỉ camera thấy, mô hình LiDAR không có box ở đây (detector {score:.2f}, sau gộp {obj.score:.2f})",
            )
        )

    conflict = False
    if obj.alternatives:
        alt, alt_score = max(obj.alternatives.items(), key=lambda kv: kv[1])
        if alt_score >= cfg.class_conflict_ratio * score:
            conflict = True
            issues.append(
                QAIssue(
                    code="CLASS_CONFLICT",
                    group="detection",
                    message=f"Cùng vị trí còn được nhận là '{alt}' (score {alt_score:.2f} so với {score:.2f})",
                )
            )

    term = (1.0 - score) + (config.qa.risk.class_conflict_penalty if conflict else 0.0)
    return {"issues": issues, "term": min(1.0, term)}


def confidence_node(state: QAState) -> dict:
    config = state["config"]
    return {"confidence": {o.object_id: check_confidence(o, config) for o in state.get("objects", [])}}
