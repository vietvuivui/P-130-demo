import hashlib
import json

import pytest

from src.models.schemas import ReviewActionRequest, ReviewState
from src.services import qc, review
from src.services.exporter import NotReadyError, export_dataset
from src.services.store import WorkspaceStore
from tests.conftest import make_frame, make_object


def decide(store, frame, config, **kw):
    """Như route /actions: áp thao tác và ghi correction log."""
    store.append_corrections([review.apply_action(frame, ReviewActionRequest(**kw), "an", config)])


def approve_all(store, frame, config):
    store.append_corrections(review.approve_low_risk(frame, "an"))
    for o in frame.objects:
        if o.review.status == "pending":
            decide(store, frame, config, action="KEEP", object_id=o.object_id)
    review.approve_frame(frame, "an", 5)
    store.save_frame(frame)


@pytest.fixture
def ws(tmp_path, config):
    s = WorkspaceStore(tmp_path / "ws")
    approve_all(s, make_frame("scene-0001_000"), config)
    s.save_frame(make_frame("scene-0001_001"))  # chưa duyệt
    return s


def check(report, cid):
    return next(c for c in report["checks"] if c["id"] == cid)


def test_log_integrity(ws, config):
    rep = qc.dataset_report(ws, config)
    assert check(rep, "audit_trail")["ok"] and rep["log_findings"] == []

    # Nhãn bị sửa thẳng trong file (không qua thao tác có log) -> log không khớp
    f = ws.load_frame("scene-0001_000")
    f.objects[0].review.final_label = "truck"
    # Object được đánh dấu duyệt mà không có dòng log nào
    f.objects.append(
        make_object("9", [1200, 400, 1300, 480]).model_copy(
            update={
                "review": ReviewState(
                    status="approved", action="KEEP", final_label="car", final_bbox=[1200, 400, 1300, 480]
                )
            }
        )
    )
    # Máy tự xoá theo keyframe (lan truyền) thì không cần log
    f.objects.append(
        make_object("10", [1300, 400, 1400, 480]).model_copy(
            update={"review": ReviewState(status="deleted", action="PROPAGATED_DELETE")}
        )
    )
    ws.save_frame(f)
    rep = qc.dataset_report(ws, config)
    assert {(e["object_id"], e["code"]) for e in rep["log_findings"]} == {("1", "LOG_MISMATCH"), ("9", "LOG_MISSING")}
    assert not check(rep, "audit_trail")["ok"] and rep["status"] == "NOT_READY"


def test_track_class_consistency(tmp_path, config):
    s = WorkspaceStore(tmp_path / "ws")
    for i, label in enumerate(("car", "truck")):
        obj = make_object("1", [100, 400, 300, 520]).model_copy(update={"track_id": "k0:1"})
        f = make_frame(f"scene-0001_00{i}", objects=[obj])
        if label != "car":
            decide(s, f, config, action="CHANGE_CLASS", object_id="1", label=label)
        approve_all(s, f, config)
    rep = qc.dataset_report(s, config)
    [t] = rep["track_findings"]
    assert t["track_id"] == "k0:1" and set(t["classes"]) == {"car", "truck"}
    assert not check(rep, "track_consistency")["ok"]


def test_ready_after_audit_and_export_manifest(ws, config):
    rep = qc.dataset_report(ws, config)
    assert rep["status"] == "NOT_READY" and rep["coverage"] == 0.5
    # Nhãn, track, log đều ổn; chỉ còn thiếu audit (2 object duyệt theo lô, 1 frame đã approve chưa ai audit)
    assert [c["id"] for c in rep["checks"] if not c["ok"]] == ["audit_objects", "audit_frames"]
    with pytest.raises(NotReadyError, match="Audit ngẫu nhiên object"):
        export_dataset(ws, config, require_ready=True)

    for kind in ("object", "frame"):
        for item in qc.create_sample(ws, config, kind, seed=0):
            qc.record_result(ws, config, item.audit_id, "ok", "", "an")
    rep = qc.dataset_report(ws, config)
    assert rep["status"] == "READY", rep["checks"]

    res = export_dataset(ws, config, require_ready=True)
    assert res.release_status == "READY" and res.frames_skipped_qc == []
    out = ws.exports_dir / res.export_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "READY"
    assert manifest["provenance"]["human_action"] == {"BATCH_APPROVE": 2, "KEEP": 1}
    # SHA256 trong manifest khớp file thật
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    assert set(manifest["files_sha256"]) == set(res.files) - {"manifest.json"}
    assert "**READY**" in (out / "qa_report.md").read_text(encoding="utf-8")
    events = [json.loads(line) for line in (out / "qc_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {e["event"] for e in events} == {"AUDIT_SAMPLE", "AUDIT_RESULT"}


def test_export_skips_frames_with_open_findings(ws, config):
    f = make_frame("scene-0001_002")
    approve_all(ws, f, config)
    f.objects[0].review.final_bbox = [1500, 400, 1700, 520]  # box ra ngoài ảnh: lỗi phải sửa
    ws.save_frame(f)
    res = export_dataset(ws, config)
    assert res.n_frames == 1 and res.frames_skipped_qc == ["scene-0001_002"]
    rep = qc.dataset_report(ws, config)
    assert rep["frames_with_open_findings"] == ["scene-0001_002"] and rep["open_by_code"] == {"BOX_OUT_OF_IMAGE": 1}
