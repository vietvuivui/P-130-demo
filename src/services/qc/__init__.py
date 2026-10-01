"""QC (kiểm soát chất lượng) nhãn — ba luồng:

1. Quick Check (`quick_check.py`): kiểm nhanh file nhãn từ ngoài, không ghi gì.
2. QC nhãn cuối (`checks.py`): kiểm lại nhãn sau khi người sửa; chặn approve frame khi còn lỗi.
3. Audit + phát hành (`audit.py`, `report.py`): audit ngẫu nhiên phần duyệt theo lô, checklist READY, báo cáo QA.
"""

from src.services.qc.audit import AuditError, audit_summary, create_sample, record_result, wilson_interval
from src.services.qc.checks import QCError, ack_finding, frame_findings, open_findings
from src.services.qc.quick_check import quick_check
from src.services.qc.report import collect, dataset_report, load_frame_findings, render_qa_report

__all__ = [
    "AuditError",
    "QCError",
    "ack_finding",
    "audit_summary",
    "collect",
    "create_sample",
    "dataset_report",
    "frame_findings",
    "load_frame_findings",
    "open_findings",
    "quick_check",
    "record_result",
    "render_qa_report",
    "wilson_interval",
]
