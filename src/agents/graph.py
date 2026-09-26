"""QA Agent (cross-modal verification) dựng bằng LangGraph.

            ┌─ confidence_check (3.1) ─┐
    START ──┼─ lidar_check      (3.2) ─┼─> issue_generation (3.4) ─> risk_scoring (3.5) ─> END
            └─ temporal_check   (3.3) ─┘

Agent hoàn toàn deterministic: không gọi LLM, chạy được trong batch và trong test.
"""

from langgraph.graph import END, START, StateGraph

from src.agents.nodes.confidence import confidence_node
from src.agents.nodes.issues import issue_node
from src.agents.nodes.lidar import lidar_node
from src.agents.nodes.risk import risk_node
from src.agents.nodes.temporal import temporal_node
from src.agents.state import QAState

CHECKS = ("confidence_check", "lidar_check", "temporal_check")


def build_graph():
    graph = StateGraph(QAState)

    graph.add_node("confidence_check", confidence_node)
    graph.add_node("lidar_check", lidar_node)
    graph.add_node("temporal_check", temporal_node)
    graph.add_node("issue_generation", issue_node)
    graph.add_node("risk_scoring", risk_node)

    for check in CHECKS:
        graph.add_edge(START, check)
    # Chờ cả ba check xong mới gộp issue
    graph.add_edge(list(CHECKS), "issue_generation")
    graph.add_edge("issue_generation", "risk_scoring")
    graph.add_edge("risk_scoring", END)

    return graph.compile()


qa_agent = build_graph()
