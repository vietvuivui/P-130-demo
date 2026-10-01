"""Tab ⚙ Cài đặt: chỉnh cài đặt trên UI (settings.json theo workspace), áp dụng lại, đánh giá trước / sau."""

import pytest

from src.api import routes
from src.services import jobs, ui_settings

S = "/api/v1/settings"


@pytest.mark.asyncio
async def test_settings_roundtrip_and_validation(client, store, config):
    f = {x["path"]: x for x in (await client.get(S)).json()["fields"]}
    assert f["propagation.flow"]["value"] == "always" and not f["propagation.flow"]["changed"]
    assert f["qa.temporal.rescore"]["default"] == "off" and "mean" in f["qa.temporal.rescore"]["choices"]

    bad = [{"qa.temporal.rescore": "max"}, {"detection.min_score": 2}, {"qa.temporal.flow": "yes"}, {"classes": 1}]
    for values in bad:
        assert (await client.put(S, json={"values": values})).status_code == 422

    r = await client.put(S, json={"values": {"qa.temporal.flow": True, "qa.temporal.rescore": "mean",
                                             "detection.min_score": "0.25", "propagation.flow": "always"}})  # fmt: skip
    f = {x["path"]: x for x in r.json()["fields"]}
    assert f["qa.temporal.rescore"]["value"] == "mean" and f["qa.temporal.rescore"]["changed"]
    assert f["detection.min_score"]["value"] == 0.25
    assert not f["propagation.flow"]["changed"]  # bằng mặc định thì không ghi vào file
    assert ui_settings.load(store.root) == {"qa.temporal.flow": True, "qa.temporal.rescore": "mean",
                                            "detection.min_score": 0.25}  # fmt: skip

    # Config của workspace = mặc định + cài đặt; config gốc không bị sửa
    cfg = ui_settings.apply(config, ui_settings.load(store.root))
    assert cfg.qa.temporal.rescore == "mean" and cfg.detection.min_score == 0.25 and config.qa.temporal.rescore == "off"

    r = await client.delete(S)
    assert not any(x["changed"] for x in r.json()["fields"]) and ui_settings.load(store.root) == {}


def test_broken_settings_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert ui_settings.load(tmp_path) == {}
    (tmp_path / "settings.json").write_text('{"detection.min_score": 5}')
    assert ui_settings.load(tmp_path) == {}


@pytest.mark.asyncio
async def test_relabel_skips_frames_people_touched(client, store):
    frame = store.load_frame("scene-0001_000")
    frame.status = "editing"
    store.save_frame(frame)
    r = await client.post("/api/v1/relabel", json={})
    assert r.status_code == 200
    done = jobs.wait(routes._job_key(store, "relabel"))
    assert done["state"] == "done", done
    assert done["result"] == {"relabeled": 0, "total": 1, "skipped": {"opened": 1, "propagated": 0, "sweep_edited": 0}}
    assert (await client.get("/api/v1/relabel")).json()["state"] == "done"


@pytest.mark.asyncio
async def test_eval_temporal_needs_ground_truth(client):
    r = await client.get("/api/v1/eval-temporal")
    assert r.json()["gt_frames"] == 0 and r.json()["last"] is None
    assert (await client.post("/api/v1/eval-temporal", json={})).status_code == 409
