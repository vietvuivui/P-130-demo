import pytest

from src.models.schemas import ReviewActionRequest
from src.services import qc, review
from src.services.store import WorkspaceStore
from tests.conftest import make_frame, make_object


def approved_frame(frame_id, config, n_low=4):
    """Frame đã approve: n_low object low-risk duyệt theo lô + 1 object high người KEEP riêng."""
    objs = [make_object(str(i), [100 * i, 400, 100 * i + 80, 480]) for i in range(1, n_low + 1)]
    objs.append(make_object("h", [900, 300, 1000, 330], label="barrier", level="high", risk=0.8, issues=("FLICKER",)))
    f = make_frame(frame_id, objects=objs)
    review.approve_low_risk(f, "tester")
    review.apply_action(f, ReviewActionRequest(action="KEEP", object_id="h"), "tester", config)
    review.approve_frame(f, "tester", 10)
    return f


@pytest.fixture
def audit_store(tmp_path, config):
    s = WorkspaceStore(tmp_path / "ws")
    for i in range(3):
        s.save_frame(approved_frame(f"scene-0001_00{i}", config))
    s.save_frame(make_frame("scene-0001_009"))  # chưa duyệt: không thuộc tổng thể audit
    return s


def test_wilson_interval():
    lo, hi = qc.wilson_interval(0, 50)
    assert lo == 0 and hi == pytest.approx(0.0713, abs=1e-4)
    assert qc.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = qc.wilson_interval(5, 10)
    assert lo == pytest.approx(1 - hi) and lo < 0.5 < hi


def test_sample_is_batch_approved_only_and_reproducible(audit_store, tmp_path, config):
    items = qc.create_sample(audit_store, config, "object", size=5, seed=7)
    assert len(items) == 5 and all(i.seed == 7 for i in items)
    # Chỉ object duyệt theo lô trong frame đã approve (không lấy #h người đã xem riêng, không lấy frame chưa duyệt)
    assert all(i.object_id != "h" and i.frame_id != "scene-0001_009" for i in items)
    assert items[0].label == "car" and items[0].bbox is not None

    # Cùng dữ liệu + cùng seed -> cùng mẫu
    other = WorkspaceStore(tmp_path / "ws2")
    for fid in audit_store.frame_ids():
        other.save_frame(audit_store.load_frame(fid))
    again = qc.create_sample(other, config, "object", size=5, seed=7)
    assert [(i.frame_id, i.object_id) for i in again] == [(i.frame_id, i.object_id) for i in items]

    # Lần lấy mẫu sau không lặp lại object đã có trong mẫu; hết tổng thể thì báo lỗi
    rest = qc.create_sample(audit_store, config, "object", size=100, seed=1)
    assert len(rest) == 12 - 5
    assert not {(i.frame_id, i.object_id) for i in rest} & {(i.frame_id, i.object_id) for i in items}
    with pytest.raises(qc.AuditError, match="Không còn"):
        qc.create_sample(audit_store, config, "object", seed=1)
    assert audit_store.qc_events()[0]["event"] == "AUDIT_SAMPLE"


def test_object_error_sends_it_back_for_review(audit_store, config):
    [item] = qc.create_sample(audit_store, config, "object", size=1, seed=3)
    qc.record_result(audit_store, config, item.audit_id, "error", "thật ra là truck", "an")

    f = audit_store.load_frame(item.frame_id)
    assert f.status == "editing"
    obj = next(o for o in f.objects if o.object_id == item.object_id)
    assert obj.review.status == "pending" and obj.qa.level == "high"
    assert obj.qa.issues[-1].code == "AUDIT_FAILED" and "truck" in obj.qa.issues[-1].message
    with pytest.raises(review.ReviewError):
        review.approve_frame(f, "an", 1)  # phải quyết định lại object đó

    with pytest.raises(qc.AuditError, match="đã có kết quả"):
        qc.record_result(audit_store, config, item.audit_id, "ok", "", "an")
    with pytest.raises(qc.AuditError) as e:
        qc.record_result(audit_store, config, "a99999", "ok", "", "an")
    assert e.value.status == 404
    assert audit_store.qc_events()[-1]["result"] == "error"


def test_frame_error_needs_missing_object_resolved(audit_store, config):
    [item] = qc.create_sample(audit_store, config, "frame", size=1, seed=3)
    qc.record_result(audit_store, config, item.audit_id, "error", "", "an")
    f = audit_store.load_frame(item.frame_id)
    assert f.status == "editing"
    [finding] = qc.open_findings(qc.frame_findings(f, config, None))
    assert finding.code == "AUDIT_MISSING_OBJECT" and item.audit_id in finding.key


def test_summary_rules(audit_store, config):
    s = qc.audit_summary(audit_store, config)
    assert s["object"]["population"] == 12 and s["frame"]["population"] == 3
    assert not s["object"]["passed"] and "Cần ít nhất" in s["object"]["note"]

    # Audit hết 3 frame, đều đúng -> đã audit toàn bộ tổng thể: đạt dù chưa đủ 10 mẫu
    for item in qc.create_sample(audit_store, config, "frame", seed=1):
        qc.record_result(audit_store, config, item.audit_id, "ok", "", "an")
    s = qc.audit_summary(audit_store, config)["frame"]
    assert s["census"] and s["passed"] and s["n"] == 3 and s["error_rate"] == 0

    # Mẫu đủ cỡ: đạt khi cận trên Wilson <= ngưỡng
    cfg = config.model_copy(deep=True)
    cfg.qc.audit.object_sample_size = 4
    cfg.qc.audit.object_max_error_upper = 0.6
    items = qc.create_sample(audit_store, cfg, "object", seed=2)
    for n, item in enumerate(items):
        qc.record_result(audit_store, cfg, item.audit_id, "error" if n == 0 else "ok", "", "an")
    s = qc.audit_summary(audit_store, cfg)["object"]
    assert s["n"] == 4 and s["errors"] == 1 and s["error_rate"] == 0.25
    assert s["ci95"][1] == pytest.approx(qc.wilson_interval(1, 4)[1], abs=1e-4)
    assert s["passed"] == (s["ci95"][1] <= 0.6)

    cfg.qc.audit.object_sample_size = 0
    assert qc.audit_summary(audit_store, cfg)["object"]["passed"]
