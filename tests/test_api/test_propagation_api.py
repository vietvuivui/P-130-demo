import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api import routes
from src.main import app
from src.services import review
from src.services.store import WorkspaceStore
from tests.test_services.test_propagation import FakeSource, fid, make_frame


@pytest_asyncio.fixture
async def seq_client(tmp_path, config):
    source = FakeSource()
    store = WorkspaceStore(tmp_path / "ws")
    for k in range(3):
        store.save_frame(make_frame(k, source))
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_config] = lambda: config
    app.dependency_overrides[routes.get_sequence_source] = lambda: source
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_propagate_endpoint(seq_client):
    r = await seq_client.post(f"/api/v1/frames/{fid(0)}/propagate", json={})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "NOT_APPROVED"

    await seq_client.post(
        f"/api/v1/frames/{fid(0)}/actions", json={"action": "CHANGE_CLASS", "object_id": "1", "label": "truck"}
    )
    await seq_client.post(f"/api/v1/frames/{fid(0)}/approve-low-risk", json={})
    await seq_client.post(f"/api/v1/frames/{fid(0)}/approve", json={"review_time_s": 12})

    r = await seq_client.post(f"/api/v1/frames/{fid(0)}/propagate", json={"max_frames": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["frames_updated"] == [fid(1)] and body["objects_propagated"] == 2

    frames = {f["frame_id"]: f for f in (await seq_client.get("/api/v1/frames")).json()}
    assert frames[fid(1)]["propagated_from"] == fid(0) and frames[fid(2)]["propagated_from"] is None

    f1 = (await seq_client.get(f"/api/v1/frames/{fid(1)}")).json()
    truck = next(o for o in f1["objects"] if o["label"] == "truck")
    assert truck["source"] == "propagated" and truck["propagation"]["keyframe_id"] == fid(0)

    cfg = (await seq_client.get("/api/v1/config")).json()
    assert "PROP_COASTING" in cfg["issue_help"]
    assert (await seq_client.post("/api/v1/frames/nope/propagate", json={})).status_code == 404


@pytest.mark.asyncio
async def test_propagate_nuscenes_scene_without_dataset_returns_503(tmp_path, config, monkeypatch):
    store = WorkspaceStore(tmp_path / "ws")
    frame = make_frame(0, FakeSource())
    review.approve_low_risk(frame, "an")
    review.approve_frame(frame, "an", 10)
    store.save_frame(frame)
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_config] = lambda: config
    monkeypatch.setattr(routes.get_settings(), "nuscenes_dataroot", str(tmp_path / "missing"))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post(f"/api/v1/frames/{fid(0)}/propagate", json={})
        assert r.status_code == 503 and r.json()["detail"]["code"] == "DATA_UNAVAILABLE"
    finally:
        app.dependency_overrides.clear()
