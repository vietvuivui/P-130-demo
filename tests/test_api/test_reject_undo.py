"""FR-15 reject kèm lý do và FR-09 undo / redo, cho frame 2D và 3D."""

import pytest

from tests.test_api.test_routes3d import store3d  # noqa: F401 — fixture

F = "/api/v1/frames/scene-0001_000"
F3 = "/api/v1/3d/frames/pointpillars/scene-0035_000"


@pytest.mark.asyncio
async def test_reject_requires_reason_and_blocks_export(client, store):
    r = await client.post(f"{F}/reject", json={"reason": "  "})
    assert r.status_code == 422
    for oid in ("1", "2", "3"):
        await client.post(f"{F}/actions", json={"action": "KEEP", "object_id": oid})
    assert (await client.post(f"{F}/approve", json={"review_time_s": 12})).json()["status"] == "approved"
    r = await client.post(f"{F}/reject", json={"reason": "Sót xe  bị che bên trái", "reviewer": "lead"})
    f = r.json()
    assert f["status"] == "rejected" and f["reject_reason"] == "Sót xe bị che bên trái" and f["rejected_by"] == "lead"
    assert f["approved_at"] is None
    # frame bị trả lại lên đầu hàng đợi, không được xuất
    assert (await client.get("/api/v1/frames", params={"status": "rejected"})).json()[0]["frame_id"] == "scene-0001_000"
    assert (await client.post("/api/v1/export")).status_code == 409
    # sửa lại -> editing, vẫn giữ lý do để người gán nhãn thấy; approve lại được
    f = (await client.post(f"{F}/actions", json={"action": "DELETE", "object_id": "3"})).json()
    assert f["status"] == "editing" and f["reject_reason"]
    assert (await client.post(f"{F}/approve", json={})).json()["status"] == "approved"
    types = [e["type"] for e in store.events()]
    assert types == ["approve", "reject", "approve"]


@pytest.mark.asyncio
async def test_undo_redo_2d(client, store):
    assert (await client.post(f"{F}/undo", json={})).status_code == 409  # chưa có thao tác
    await client.post(f"{F}/actions", json={"action": "DELETE", "object_id": "3"})
    await client.post(f"{F}/actions", json={"action": "ADD_BOX", "bbox": [10, 10, 60, 60], "label": "car"})
    assert (await client.get(f"{F}/history")).json() == {"undo": 2, "redo": 0}
    f = (await client.post(f"{F}/undo", json={})).json()
    assert [o["object_id"] for o in f["objects"]] == ["1", "2", "3"]  # box vẽ thêm biến mất
    f = (await client.post(f"{F}/undo", json={})).json()
    assert next(o for o in f["objects"] if o["object_id"] == "3")["review"]["status"] == "pending"
    f = (await client.post(f"{F}/redo", json={})).json()
    assert next(o for o in f["objects"] if o["object_id"] == "3")["review"]["status"] == "deleted"
    assert (await client.get(f"{F}/history")).json() == {"undo": 1, "redo": 1}
    # thao tác mới xoá ngăn redo
    await client.post(f"{F}/actions", json={"action": "KEEP", "object_id": "1"})
    assert (await client.get(f"{F}/history")).json() == {"undo": 2, "redo": 0}
    actions = [e["human_action"] for e in store.corrections()]
    assert actions[-3:] == ["UNDO", "REDO", "KEEP"] and "UNDO" in actions
    # frame đã approve: phải mở lại trước khi undo
    await client.post(f"{F}/actions", json={"action": "KEEP", "object_id": "2"})
    await client.post(f"{F}/approve", json={})
    assert (await client.post(f"{F}/undo", json={})).status_code == 409


@pytest.mark.asyncio
async def test_reject_and_undo_3d(client, store3d):  # noqa: F811
    assert (await client.post(f"{F3}/reject", json={"reason": "ok"})).status_code == 422  # lý do quá ngắn
    f = (await client.post(f"{F3}/reject", json={"reason": "Box 2 lệch khỏi xe"})).json()
    assert f["status"] == "rejected"
    box = {"center": [11.0, 0.5, 0.0], "size": [2.0, 4.6, 1.6], "yaw": 0.3}
    f = (await client.post(f"{F3}/actions", json={"action": "EDIT_BOX", "object_id": "1", "box": box})).json()
    assert f["status"] == "editing" and f["objects"][0]["box"]["center"] == [11.0, 0.5, 0.0]
    f = (await client.post(f"{F3}/undo", json={})).json()
    assert f["objects"][0]["box"]["center"] == [10.0, 0.0, 0.0] and f["status"] == "rejected"
    f = (await client.post(f"{F3}/redo", json={})).json()
    assert f["objects"][0]["box"]["center"] == [11.0, 0.5, 0.0]
    m = (await client.get("/api/v1/3d/metrics", params={"model": "pointpillars"})).json()
    assert m["rejected"] == 0  # đã sửa lại -> editing
