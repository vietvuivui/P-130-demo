import numpy as np
import pytest

from src.services.evaluation import average_precision, evaluate_map, evaluate_qa
from tests.conftest import make_frame, make_object


def gt(bbox, label="car", ignore=False):
    return {"bbox": bbox, "label": label, "ignore": ignore}


def test_average_precision():
    assert average_precision(np.array([1, 1], dtype=float), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0, 1], dtype=float), 1) == pytest.approx(0.5)
    assert average_precision(np.array([], dtype=float), 3) == 0.0


def test_map_ignores_low_visibility_gt():
    f = make_frame(
        objects=[
            make_object("1", [0, 0, 100, 100], score=0.9),
            make_object("2", [300, 0, 400, 100], score=0.8),  # trùng GT ignore -> không tính FP
            make_object("3", [600, 0, 700, 100], score=0.7),  # FP
        ]
    )
    res = evaluate_map([f], {f.frame_id: [gt([0, 0, 100, 100]), gt([300, 0, 400, 100], ignore=True)]}, 0.5)
    assert res["per_class"]["car"]["n_gt"] == 1
    assert res["per_class"]["car"]["ap"] == pytest.approx(1.0)


def test_qa_flags_vs_gt():
    f = make_frame(
        objects=[
            make_object("1", [0, 0, 100, 100], level="low"),  # đúng, không cờ
            make_object("2", [300, 0, 400, 100], level="high", issues=("FLICKER",)),  # FP, có cờ
            make_object("3", [600, 0, 700, 100], label="truck", level="low"),  # sai lớp, lọt cờ
            make_object("r1", [900, 0, 1000, 100], source="track", level="medium", issues=("RECOVERED_BY_TRACK",)),
        ]
    )
    gts = [gt([0, 0, 100, 100]), gt([600, 0, 700, 100]), gt([900, 0, 1000, 100])]
    res = evaluate_qa([f], {f.frame_id: gts})
    assert res["objects_evaluated"] == 3 and res["objects_need_fix"] == 2
    assert res["flag_recall"] == 0.5 and res["flag_precision"] == 1.0
    assert res["low_risk_error_rate"] == 0.5
    assert res["by_issue"]["FLICKER"] == {"n": 1, "needs_fix": 1, "precision": 1.0}
    assert res["fn_total"] == 1 and res["fn_recovered_by_track"] == 1
