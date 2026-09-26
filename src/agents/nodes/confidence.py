"""3.1 Confidence check: score thấp và xung đột lớp."""

from __future__ import annotations

from src.agents.state import QAState
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import LabelObject, QAIssue


def check_confidence(obj: LabelObject, config: AutoLabelConfig) -> dict:
    cfg = config.qa.confidence
    issues = []
    if obj.score < cfg.low_threshold:
        issues.append(
            QAIssue(
                code="LOW_CONFIDENCE", group="detection", message=f"Score {obj.score:.2f} < {cfg.low_threshold:.2f}"
            )
        )

    conflict = False
    if obj.alternatives:
        alt, alt_score = max(obj.alternatives.items(), key=lambda kv: kv[1])
        if alt_score >= cfg.class_conflict_ratio * obj.score:
            conflict = True
            issues.append(
                QAIssue(
                    code="CLASS_CONFLICT",
                    group="detection",
                    message=f"Cùng vị trí còn được nhận là '{alt}' (score {alt_score:.2f} so với {obj.score:.2f})",
                )
            )

    term = (1.0 - obj.score) + (config.qa.risk.class_conflict_penalty if conflict else 0.0)
    return {"issues": issues, "term": min(1.0, term)}


def confidence_node(state: QAState) -> dict:
    config = state["config"]
    return {"confidence": {o.object_id: check_confidence(o, config) for o in state.get("objects", [])}}
