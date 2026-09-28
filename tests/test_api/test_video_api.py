import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src import demo
from src.api import routes
from src.main import app
from src.services.detectors import DetectorEnsemble
from src.services.store import WorkspaceStore


@pytest.fixture(scope="module")
def demo_mp4(tmp_path_factory):
    return demo.render_video(tmp_path_factory.mktemp("video") / "street.mp4", seconds=2)


@pytest_asyncio.fixture
async def video_client(tmp_path, config):
    config.detection.detectors = ["demo"]
    store = WorkspaceStore(tmp_path / "ws")
    ensemble = DetectorEnsemble(config, store.root / "cache" / "detections")
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_config] = lambda: config
    app.dependency_overrides[routes.get_ensemble] = lambda: ensemble
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_then_review_and_propagate(video_client, demo_mp4):
    with open(demo_mp4, "rb") as fh:
        r = await video_client.post("/api/v1/videos/upload", files={"file": ("street.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    vid = r.json()["video_id"]
    assert r.json()["source"] == "upload"

    # Auto-label chạy bằng BackgroundTasks; ASGITransport đợi task nền xong mới trả về
    detail = (await video_client.get(f"/api/v1/videos/{vid}")).json()
    assert detail["status"] == "ready" and detail["n_frames"] == 4
    assert [f["t"] for f in detail["frames"]] == [0.0, 0.5, 1.0, 1.5]
    assert (await video_client.get("/api/v1/videos")).json()[0]["video_id"] == vid

    f0 = detail["frames"][0]["frame_id"]
    img = await video_client.get(f"/api/v1/frames/{f0}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    await video_client.post(f"/api/v1/frames/{f0}/approve-low-risk", json={})
    frame = (await video_client.get(f"/api/v1/frames/{f0}")).json()
    for o in frame["objects"]:
        if o["review"]["status"] == "pending":
            await video_client.post(
                f"/api/v1/frames/{f0}/actions", json={"action": "KEEP", "object_id": o["object_id"]}
            )
    assert (await video_client.post(f"/api/v1/frames/{f0}/approve", json={"review_time_s": 8})).status_code == 200

    r = await video_client.post(f"/api/v1/frames/{f0}/propagate", json={})
    assert r.status_code == 200, r.text
    assert len(r.json()["frames_updated"]) == 3
    detail = (await video_client.get(f"/api/v1/videos/{vid}")).json()
    assert detail["approved"] == 1 and detail["propagated"] == 3


@pytest.mark.asyncio
async def test_upload_rejects_other_formats(video_client, tmp_path):
    r = await video_client.post("/api/v1/videos/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "VIDEO_FORMAT"
    r = await video_client.post("/api/v1/videos/upload", files={"file": ("broken.mp4", b"hello", "video/mp4")})
    assert r.status_code == 422 and r.json()["detail"]["code"] in {"VIDEO_UNREADABLE", "VIDEO_EMPTY"}
    assert (await video_client.get("/api/v1/videos")).json() == []
    assert (await video_client.get("/api/v1/videos/nope")).status_code == 404


@pytest.mark.asyncio
async def test_upload_over_size_limit_is_rejected(video_client, config, demo_mp4):
    config.video.max_upload_mb = 0
    with open(demo_mp4, "rb") as fh:
        r = await video_client.post("/api/v1/videos/upload", files={"file": ("street.mp4", fh, "video/mp4")})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "VIDEO_TOO_LARGE"
    assert (await video_client.get("/api/v1/videos")).json() == []
