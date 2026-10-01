import pytest

FID = "scene-0001_000"


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_config_and_status(client):
    cfg = (await client.get("/api/v1/config")).json()
    assert "car" in cfg["classes"] and "NO_LIDAR_SUPPORT" in cfg["issue_help"]
    assert (await client.get("/api/v1/status")).json()["frames"] == 1


@pytest.mark.asyncio
async def test_list_and_get_frame(client):
    frames = (await client.get("/api/v1/frames")).json()
    assert frames[0]["frame_id"] == FID
    assert frames[0]["counts"] == {"low": 2, "medium": 0, "high": 1}
    assert frames[0]["pending"] == 3

    frame = (await client.get(f"/api/v1/frames/{FID}")).json()
    assert len(frame["objects"]) == 3

    assert (await client.get("/api/v1/frames/nope")).status_code == 404
    assert (await client.get("/api/v1/frames/..%2Fsecret")).status_code in (400, 404)


@pytest.mark.asyncio
async def test_image_endpoint(client):
    r = await client.get(f"/api/v1/frames/{FID}/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    # Sweep có trong record nhưng file không tồn tại trong dataroot tạm
    assert (await client.get(f"/api/v1/frames/{FID}/image?offset=-1")).status_code == 404
    assert (await client.get(f"/api/v1/frames/{FID}/image?offset=2")).json()["detail"]["code"] == "SWEEP_NOT_FOUND"


@pytest.mark.asyncio
async def test_review_flow_and_export(client):
    r = await client.post("/api/v1/export")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "NOTHING_TO_EXPORT"

    r = await client.post(f"/api/v1/frames/{FID}/approve-low-risk", json={"reviewer": "an"})
    assert r.status_code == 200 and r.json()["status"] == "editing"

    r = await client.post(f"/api/v1/frames/{FID}/approve", json={})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PENDING_OBJECTS"

    r = await client.post(
        f"/api/v1/frames/{FID}/actions", json={"action": "CHANGE_CLASS", "object_id": "3", "label": "unicorn"}
    )
    assert r.status_code == 422

    r = await client.post(
        f"/api/v1/frames/{FID}/actions",
        json={"action": "CHANGE_CLASS", "object_id": "3", "label": "traffic_cone", "reviewer": "an"},
    )
    assert r.status_code == 200

    # QC nhãn cuối: box 100x30 px đổi thành traffic_cone -> tỉ lệ rộng/cao sai với lớp mới, chặn approve
    r = await client.post(f"/api/v1/frames/{FID}/approve", json={"reviewer": "an", "review_time_s": 42})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "QC_FINDINGS"
    [finding] = r.json()["detail"]["findings"]
    assert finding["code"] == "ASPECT_RATIO_ABNORMAL" and finding["object_id"] == "3"
    ack = {"key": finding["key"], "fingerprint": finding["fingerprint"], "note": "cone nằm ngang", "reviewer": "an"}
    assert (await client.post(f"/api/v1/frames/{FID}/qc/ack", json=ack)).json()["open"] == 0

    r = await client.post(f"/api/v1/frames/{FID}/approve", json={"reviewer": "an", "review_time_s": 42})
    assert r.json()["status"] == "approved"

    log = (await client.get("/api/v1/corrections", params={"frame_id": FID})).json()
    assert [e["human_action"] for e in log] == ["CHANGE_CLASS", "BATCH_APPROVE", "BATCH_APPROVE"]
    assert log[0]["final_class"] == "traffic_cone" and log[0]["reviewer"] == "an"

    m = (await client.get("/api/v1/metrics")).json()
    assert m["frames"]["approved"] == 1 and m["m4_correction_rate"] == pytest.approx(1 / 3, abs=1e-3)

    exp = (await client.post("/api/v1/export")).json()
    assert exp["n_frames"] == 1 and exp["n_objects"] == 3
    assert (await client.get("/api/v1/exports")).json() == [exp["export_id"]]
    r = await client.get(f"/api/v1/exports/{exp['export_id']}/coco.json")
    assert r.status_code == 200
    assert (await client.get(f"/api/v1/exports/{exp['export_id']}/secret.txt")).status_code == 404


@pytest.mark.asyncio
async def test_ui_is_served(client):
    r = await client.get("/ui/")
    assert r.status_code == 200 and "AutoLabel 3D" in r.text
