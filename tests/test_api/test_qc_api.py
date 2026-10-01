import json

import pytest

FID = "scene-0001_000"


async def decide_all(client):
    await client.post(f"/api/v1/frames/{FID}/approve-low-risk", json={"reviewer": "an"})
    await client.post(f"/api/v1/frames/{FID}/actions", json={"action": "KEEP", "object_id": "3", "reviewer": "an"})


@pytest.mark.asyncio
async def test_approve_gate_and_ack(client):
    await decide_all(client)
    # Vẽ trùng lên #1 -> box trùng: QC chặn approve cho tới khi xử lý
    r = await client.post(
        f"/api/v1/frames/{FID}/actions", json={"action": "ADD_BOX", "bbox": [101, 400, 300, 521], "label": "car"}
    )
    assert r.status_code == 200
    qc = (await client.get(f"/api/v1/frames/{FID}/qc")).json()
    assert qc["open"] == 1 and qc["findings"][0]["code"] == "DUPLICATE_BOX"
    r = await client.post(f"/api/v1/frames/{FID}/approve", json={})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "QC_FINDINGS"

    dup = qc["findings"][0]
    bad = {"key": dup["key"], "fingerprint": "stale"}
    assert (await client.post(f"/api/v1/frames/{FID}/qc/ack", json=bad)).json()["detail"]["code"] == "QC_ACK_REJECTED"

    # Làm theo đề xuất (xoá box yếu hơn) thì hết lỗi, approve được
    s = dup["suggestion"]
    await client.post(f"/api/v1/frames/{FID}/actions", json={"action": s["action"], "object_id": s["object_id"]})
    assert (await client.get(f"/api/v1/frames/{FID}/qc")).json()["open"] == 0
    assert (await client.post(f"/api/v1/frames/{FID}/approve", json={})).json()["status"] == "approved"


@pytest.mark.asyncio
async def test_ack_is_logged(client, store):
    await decide_all(client)
    await client.post(
        f"/api/v1/frames/{FID}/actions", json={"action": "CHANGE_CLASS", "object_id": "3", "label": "traffic_cone"}
    )
    [f] = (await client.get(f"/api/v1/frames/{FID}/qc")).json()["findings"]
    body = {"key": f["key"], "fingerprint": f["fingerprint"], "note": "cone đổ nằm ngang", "reviewer": "an"}
    r = await client.post(f"/api/v1/frames/{FID}/qc/ack", json=body)
    assert r.json()["open"] == 0 and r.json()["findings"][0]["acked"]
    [event] = store.qc_events(FID)
    assert event["event"] == "QC_ACK" and event["code"] == "ASPECT_RATIO_ABNORMAL" and event["note"] == body["note"]


@pytest.mark.asyncio
async def test_quick_check_upload(client):
    labels = {"frames": [{"frame_id": FID, "objects": [{"bbox": [700, 420, 760, 560], "class": "pedestrian"}]}]}
    files = {"file": ("vendor.json", json.dumps(labels).encode(), "application/json")}
    r = await client.post("/api/v1/qc/quick-check", files=files)
    assert r.status_code == 200
    res = r.json()
    assert res["format"] == "frames" and res["n_labels"] == 1 and res["by_code"] == {"POSSIBLY_MISSING": 1}
    # Không ghi gì vào workspace
    assert (await client.get(f"/api/v1/frames/{FID}")).json()["status"] == "auto"

    r = await client.post("/api/v1/qc/quick-check", files={"file": ("x.txt", b"1 2 3", "text/plain")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "QC_FORMAT"
    r = await client.post("/api/v1/qc/quick-check", files={"file": ("x.json", b"{broken", "application/json")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "QC_PARSE"


@pytest.mark.asyncio
async def test_audit_flow_and_release_gate(client):
    r = await client.post("/api/v1/qc/audit/sample", json={"kind": "object"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "AUDIT_EMPTY_POOL"  # chưa có gì duyệt theo lô

    await decide_all(client)
    await client.post(f"/api/v1/frames/{FID}/approve", json={})
    r = await client.post("/api/v1/export", params={"require_ready": True})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "NOT_READY"

    state = (await client.post("/api/v1/qc/audit/sample", json={"kind": "object", "seed": 1})).json()
    assert len(state["items"]) == 2 and state["summary"]["object"]["pending"] == 2
    first, second = state["items"]
    await client.post(f"/api/v1/qc/audit/{first['audit_id']}", json={"result": "ok"})
    r = await client.post(f"/api/v1/qc/audit/{first['audit_id']}", json={"result": "ok"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "AUDIT_DONE"
    assert (await client.post("/api/v1/qc/audit/a99999", json={"result": "ok"})).status_code == 404

    # Mẫu sai -> frame mở lại, object quay về chờ duyệt
    state = (
        await client.post(f"/api/v1/qc/audit/{second['audit_id']}", json={"result": "error", "note": "sai lớp"})
    ).json()
    assert state["summary"]["object"]["errors"] == 1
    frame = (await client.get(f"/api/v1/frames/{FID}")).json()
    obj = next(o for o in frame["objects"] if o["object_id"] == second["object_id"])
    assert frame["status"] == "editing" and obj["review"]["status"] == "pending" and obj["qa"]["level"] == "high"
    report = (await client.get("/api/v1/qc/report")).json()
    assert report["status"] == "NOT_READY" and report["frames"]["approved"] == 0
