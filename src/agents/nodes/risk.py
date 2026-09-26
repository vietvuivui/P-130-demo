"""3.5 Risk scoring: risk = w1 (1 - score) + w2 lidar + w3 temporal + w4 geometric."""

from __future__ import annotations

from src.agents.state import QAState
from src.models.qa_config import RiskCfg
from src.models.schemas import QAResult, RiskLevel


def risk_level(risk: float, cfg: RiskCfg) -> RiskLevel:
    if risk >= cfg.levels.high:
        return "high"
    if risk >= cfg.levels.medium:
        return "medium"
    return "low"


def score_risk(qa: QAResult, cfg: RiskCfg) -> QAResult:
    w = cfg.weights.model_dump()
    risk = sum(w[k] * qa.terms.get(k, 0.0) for k in w) / sum(w.values())
    # Có issue thì không bao giờ rơi vào nhóm duyệt theo lô
    if qa.issues:
        risk = max(risk, cfg.issue_floor)
    risk = round(min(1.0, risk), 3)
    return qa.model_copy(update={"risk": risk, "level": risk_level(risk, cfg)})


def risk_node(state: QAState) -> dict:
    cfg = state["config"].qa.risk
    reviewed = [o.model_copy(update={"qa": score_risk(o.qa, cfg)}) for o in state.get("reviewed", [])]
    reviewed.sort(key=lambda o: -o.qa.risk)
    return {"reviewed": reviewed, "frame_risk": max((o.qa.risk for o in reviewed), default=0.0)}
