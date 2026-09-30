import json

import pytest

from src.models.schemas import ReviewActionRequest
from src.services import review
from src.services.exporter import NothingToExportError, export_dataset
from tests.conftest import make_frame


def act(frame, config, **kw):
    return review.apply_action(frame, ReviewActionRequest(**kw), "tester", config)


def test_keep_delete_change_class(config):
    f = make_frame()
    entry = act(f, config, action="CHANGE_CLASS", object_id="3", label="traffic_cone")
    assert f.status == "editing"
    assert entry["human_action"] == "CHANGE_CLASS"
    assert entry["prediction"]["class"] == "barrier" and entry["final_class"] == "traffic_cone"
    assert entry["qa"]["issues"] == ["NO_LIDAR_SUPPORT", "FLICKER"]

    act(f, config, action="DELETE", object_id="2")
    objs = {o.object_id: o for o in f.objects}
    assert objs["2"].review.status == "deleted"
    assert objs["3"].review.final_label == "traffic_cone"


def test_edit_box_is_clipped_and_add_box(config):
    f = make_frame()
    act(f, config, action="EDIT_BOX", object_id="1", bbox=[-20, 400, 300, 520])
    assert f.objects[0].review.final_bbox == [0, 400, 300, 520]
    act(f, config, action="ADD_BOX", bbox=[10, 10, 50, 60], label="traffic_cone")
    added = f.objects[-1]
    assert added.source == "human" and added.object_id == "h1" and added.review.status == "approved"


def test_invalid_actions(config):
    f = make_frame()
    with pytest.raises(review.ReviewError):
        act(f, config, action="CHANGE_CLASS", object_id="1", label="unicorn")
    with pytest.raises(review.ReviewError):
        act(f, config, action="KEEP", object_id="404")
    with pytest.raises(review.ReviewError):
        act(f, config, action="ADD_BOX", bbox=[10, 10, 50, 60])


def test_approve_requires_all_decided(config):
    f = make_frame()
    entries = review.approve_low_risk(f, "tester")
    assert {e["object_id"] for e in entries} == {"1", "2"}
    assert all(e["human_action"] == "BATCH_APPROVE" for e in entries)
    with pytest.raises(review.ReviewError, match="1 object"):
        review.approve_frame(f, "tester", 12.0)
    act(f, config, action="KEEP", object_id="3")
    review.approve_frame(f, "tester", 12.0)
    assert f.status == "approved" and f.review_time_s == 12.0
    with pytest.raises(review.ReviewError):
        act(f, config, action="DELETE", object_id="1")


def test_needs_fix_m4_definition(config):
    f = make_frame()
    act(f, config, action="EDIT_BOX", object_id="1", bbox=[101, 400, 300, 520])  # IoU > 0.9
    act(f, config, action="EDIT_BOX", object_id="2", bbox=[700, 420, 740, 560])  # IoU < 0.9
    act(f, config, action="KEEP", object_id="3")
    objs = {o.object_id: o for o in f.objects}
    assert not review.needs_fix(objs["1"])
    assert review.needs_fix(objs["2"])
    assert not review.needs_fix(objs["3"])


def test_metrics(config):
    f = make_frame()
    review.approve_low_risk(f, "tester")
    act(f, config, action="DELETE", object_id="3")
    act(f, config, action="ADD_BOX", bbox=[10, 10, 50, 60], label="car")
    review.approve_frame(f, "tester", 30.0)
    m = review.compute_metrics([f, make_frame("scene-0001_001")])
    assert m["frames"] == {"total": 2, "auto": 1, "editing": 0, "approved": 1, "rejected": 0}
    assert m["model_objects_reviewed"] == 3 and m["model_objects_fixed"] == 1
    assert m["flag_precision"] == 1.0 and m["flag_recall"] == 1.0
    assert m["fix_rate_by_level"]["low"]["rate"] == 0.0
    assert m["fix_rate_by_issue"]["NO_LIDAR_SUPPORT"] == {"flagged": 1, "fixed": 1, "rate": 1.0}
    assert m["human_added_boxes"] == 1
    assert m["m1_avg_review_time_s"] == 30.0


def test_export_only_approved(store, config):
    with pytest.raises(NothingToExportError):
        export_dataset(store, config)

    f = store.load_frame("scene-0001_000")
    review.approve_low_risk(f, "tester")
    act(f, config, action="CHANGE_CLASS", object_id="3", label="traffic_cone")
    review.approve_frame(f, "tester", 5.0)
    store.save_frame(f)
    other = make_frame("scene-0001_001")
    store.save_frame(other)

    res = export_dataset(store, config)
    assert res.n_frames == 1 and res.n_objects == 3
    out = store.exports_dir / res.export_id
    coco = json.loads((out / "coco.json").read_text(encoding="utf-8"))
    assert [img["frame_id"] for img in coco["images"]] == ["scene-0001_000"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert sorted(cats[a["category_id"]] for a in coco["annotations"]) == ["car", "pedestrian", "traffic_cone"]
    assert coco["annotations"][0]["bbox"] == [100, 400, 200, 120]  # xywh
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames_skipped_not_approved"] == 1
