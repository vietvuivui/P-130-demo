import numpy as np
import pytest

from src.models.schemas import QCFinding, ReviewActionRequest, ReviewState
from src.services import qc, review
from src.services.qc.checks import FinalLabel, FrameContext, check_labels
from tests.conftest import make_frame, make_object


def act(frame, config, **kw):
    return review.apply_action(frame, ReviewActionRequest(**kw), "tester", config)


def codes(findings):
    return sorted(f.code for f in findings)


def lidar_aux(box, depth, n=40):
    """Điểm LiDAR giả: n điểm rải đều trong vùng giữa box, cùng một độ sâu."""
    x1, y1, x2, y2 = box
    rng = np.random.default_rng(0)
    u = rng.uniform(x1 + 0.3 * (x2 - x1), x2 - 0.3 * (x2 - x1), n)
    v = rng.uniform(y1 + 0.3 * (y2 - y1), y2 - 0.3 * (y2 - y1), n)
    return {"u": u.tolist(), "v": v.tolist(), "d": [depth] * n}


def test_unchanged_labels_are_not_rechecked(config):
    # #3 có NO_LIDAR_SUPPORT + FLICKER từ QA Agent; người KEEP mà không sửa -> người đã thấy issue, QC không báo lại
    f = make_frame()
    review.approve_low_risk(f, "tester")
    act(f, config, action="KEEP", object_id="3")
    assert qc.frame_findings(f, config, lidar_aux([0, 0, 10, 10], 5.0)) == []


def test_change_class_is_rechecked(config):
    f = make_frame()
    act(f, config, action="CHANGE_CLASS", object_id="3", label="traffic_cone")  # box 100x30 px
    [finding] = qc.frame_findings(f, config, None)
    assert finding.code == "ASPECT_RATIO_ABNORMAL" and finding.severity == "warning"
    assert finding.object_id == "3" and finding.key == "ASPECT_RATIO_ABNORMAL:3"
    assert finding.message.startswith("Nhãn cuối:")


def test_human_box_rechecked_against_lidar(config):
    f = make_frame()
    box = [200, 300, 400, 500]  # cao 200 px
    act(f, config, action="ADD_BOX", bbox=box, label="car")
    # Không có điểm LiDAR nào trong box cao 200 px -> box "ma"
    assert codes(qc.frame_findings(f, config, lidar_aux([1200, 600, 1300, 700], 10.0))) == ["NO_LIDAR_SUPPORT"]
    # Ở 30 m, box cao 200 px -> vật cao ~4.7 m: quá cao cho car
    assert codes(qc.frame_findings(f, config, lidar_aux(box, 30.0))) == ["SIZE_DEPTH_MISMATCH"]
    # Ở 10 m -> ~1.6 m: hợp lý
    assert qc.frame_findings(f, config, lidar_aux(box, 10.0)) == []


def test_duplicate_box_suggests_deleting_the_weaker(config):
    f = make_frame()
    act(f, config, action="ADD_BOX", bbox=[102, 401, 301, 521], label="car")  # vẽ lại đúng #1 (car đang chờ)
    [dup] = qc.frame_findings(f, config, None)
    assert dup.code == "DUPLICATE_BOX" and dup.severity == "warning"
    # Box người vẽ (h1) được giữ, box detector chưa ai xem (#1) được đề xuất xoá
    assert dup.object_id == "1" and dup.other_object_id == "h1"
    assert dup.suggestion == {"action": "DELETE", "object_id": "1"}
    assert dup.key == "DUPLICATE_BOX:1:h1"

    act(f, config, action="DELETE", object_id="1")
    assert qc.frame_findings(f, config, None) == []


def test_cross_class_overlap_and_allowed_pairs(config):
    ctx = FrameContext("f", 1600, 900)
    box = [500, 300, 600, 500]
    rider = [FinalLabel("a", "pedestrian", box, changed=False), FinalLabel("b", "bicycle", box, changed=False)]
    assert check_labels(rider, ctx, config) == []  # người trên xe đạp: nuScenes gán riêng hai nhãn

    two = [FinalLabel("a", "car", box, changed=False), FinalLabel("b", "truck", [501, 300, 600, 500], changed=False)]
    [f] = check_labels(two, ctx, config)
    assert f.code == "OVERLAP_CROSS_CLASS" and f.suggestion is None


def test_structural_errors(config):
    ctx = FrameContext("f", 1600, 900)
    labels = [
        FinalLabel("a", "unicorn", [10, 10, 50, 50], changed=False),
        FinalLabel("b", "car", [50, 50, 10, 60], changed=False),  # x2 < x1
        FinalLabel("c", "car", [1500, 100, 1700, 200], changed=False),  # vượt mép phải
        FinalLabel("d", "car", [1700, 100, 1800, 200], changed=False),  # nằm hẳn ngoài ảnh
        FinalLabel("e", "car", [float("nan"), 0, 10, 10], changed=False),
    ]
    by_id = {}
    for f in check_labels(labels, ctx, config):
        by_id.setdefault(f.object_id, []).append(f)
    assert [f.code for f in by_id["a"]] == ["UNKNOWN_CLASS"]
    assert "toạ độ ngược" in by_id["b"][0].message
    [out] = by_id["c"]
    assert out.code == "BOX_OUT_OF_IMAGE" and out.suggestion == {
        "action": "EDIT_BOX",
        "object_id": "c",
        "bbox": [1500, 100, 1600, 200],
    }
    assert by_id["d"][0].code == "INVALID_BOX" and by_id["e"][0].code == "INVALID_BOX"
    assert all(f.severity == "error" for fs in by_id.values() for f in fs)


def test_box_cut_by_image_border_is_valid(config):
    # Vật bị mép ảnh cắt còn 2.3 px (gặp thật trong GT nuScenes) không phải box hỏng
    ctx = FrameContext("f", 1600, 900)
    assert check_labels([FinalLabel("a", "barrier", [1597.7, 480.7, 1600.0, 507.5], changed=False)], ctx, config) == []


def test_ack_warning_and_invalidation(config):
    f = make_frame()
    act(f, config, action="CHANGE_CLASS", object_id="3", label="traffic_cone")
    [finding] = qc.frame_findings(f, config, None)
    qc.ack_finding(f, [finding], finding.key, finding.fingerprint, "tester", "cone nằm", "2026-09-29T00:00:00+00:00")
    [acked] = qc.frame_findings(f, config, None)
    assert acked.acked and qc.open_findings([acked]) == []

    # Sửa box sau khi đã xác nhận -> nhãn khác cái người đã xem -> phải kiểm lại
    act(f, config, action="EDIT_BOX", object_id="3", bbox=[900, 300, 1010, 330])
    [again] = qc.frame_findings(f, config, None)
    assert again.key == finding.key and again.fingerprint != finding.fingerprint and not again.acked

    with pytest.raises(qc.QCError, match="không còn"):
        qc.ack_finding(f, [again], finding.key, finding.fingerprint, "tester", "", "t")


def test_errors_cannot_be_acked(config):
    obj = make_object("1", [100, 400, 300, 520])
    obj.review = ReviewState(status="approved", action="KEEP", final_label="unicorn", final_bbox=obj.bbox)
    f = make_frame(objects=[obj])
    [err] = qc.frame_findings(f, config, None)
    assert err.code == "UNKNOWN_CLASS"
    with pytest.raises(qc.QCError, match="phải sửa"):
        qc.ack_finding(f, [err], err.key, err.fingerprint, "tester", "", "t")


def test_manual_findings_from_audit_are_resolved_by_ack(config):
    f = make_frame()
    f.qc_manual.append(
        QCFinding(
            code="AUDIT_MISSING_OBJECT",
            severity="warning",
            message="thiếu xe",
            frame_id=f.frame_id,
            key="AUDIT_MISSING_OBJECT:a00001",
            fingerprint="audit",
        )
    )
    [m] = qc.frame_findings(f, config, None)
    assert m.code == "AUDIT_MISSING_OBJECT" and not m.acked
    qc.ack_finding(f, [m], m.key, m.fingerprint, "tester", "đã vẽ thêm", "t")
    assert qc.open_findings(qc.frame_findings(f, config, None)) == []
