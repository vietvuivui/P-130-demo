"""Sửa tự do ở sweep t±n: thao tác, tính lại QA / score keyframe, hoàn tác, không lọt vào file xuất."""

import pytest

from src.models.schemas import Detection, SweepInfo
from tests.conftest import make_frame, make_object

F = "/api/v1/frames/scene-0002_000"


@pytest.fixture
def sweep_frame(store, config):
    config.qa.temporal.rescore = "mean"  # bật tính lại score theo sweep (mặc định tắt) để kiểm tra luôn phần đó
    """Keyframe có xe #1 ở cả 4 sweep, 'barrier' #2 chỉ ở keyframe (FLICKER), sweep -1 còn một box nhận nhầm."""
    car = [100.0, 400.0, 300.0, 520.0]
    frame = make_frame(
        "scene-0002_000",
        [make_object("1", car, score=0.5, level="medium", risk=0.3),
         make_object("2", [900, 300, 1000, 330], label="barrier", score=0.45, level="medium", risk=0.4)],  # fmt: skip
    )
    wrong = Detection(bbox=[600, 300, 700, 400], label="truck", score=0.6)
    frame.sweeps = [
        SweepInfo(offset=o, sd_token=f"sd{o}", path=f"sweeps/CAM_FRONT/{o}.jpg", timestamp=1_000_000 + o * 83_333,
                  detections=[Detection(bbox=car, label="car", score=0.6)] + ([wrong] if o == -1 else []))
        for o in (-2, -1, 1, 2)
    ]  # fmt: skip
    store.save_frame(frame)
    return frame


@pytest.mark.asyncio
async def test_sweep_edits_requalify_keyframe(client, store, sweep_frame):
    # Người vẽ barrier ở t-1 và t+1 (detector sót) -> barrier không còn FLICKER, score keyframe tăng
    for o in (-1, 1):
        r = await client.post(f"{F}/sweeps/{o}/actions",
                              json={"action": "ADD_BOX", "bbox": [900, 300, 1000, 330], "label": "barrier"})  # fmt: skip
        assert r.status_code == 200
    f = r.json()
    barrier = next(o for o in f["objects"] if o["object_id"] == "2")
    assert "FLICKER" not in [i["code"] for i in barrier["qa"]["issues"]]
    assert barrier["det_score"] == 0.45 and barrier["score"] == pytest.approx((0.45 + 1 + 1) / 5, abs=1e-3)
    car = next(o for o in f["objects"] if o["object_id"] == "1")
    assert car["score"] == pytest.approx((0.5 + 0.6 * 4) / 5, abs=1e-3) and car["track"]["2"] is not None
    assert f["status"] == "editing"
    sw = next(s for s in f["sweeps"] if s["offset"] == -1)
    assert [b["box_id"] for b in sw["boxes"]] == ["1", "2", "h1"] and len(sw["detections"]) == 2  # bản máy giữ nguyên

    # Xoá box nhận nhầm, đổi lớp, sửa box; box không có -> 422; lớp lạ -> 422
    base = f"{F}/sweeps/-1/actions"
    assert (await client.post(base, json={"action": "DELETE", "box_id": "2"})).status_code == 200
    assert (await client.post(base, json={"action": "DELETE", "box_id": "9"})).status_code == 422
    assert (await client.post(base, json={"action": "CHANGE_CLASS", "box_id": "1", "label": "ufo"})).status_code == 422
    f = (await client.post(base, json={"action": "EDIT_BOX", "box_id": "1", "bbox": [102, 401, 302, 521]})).json()
    boxes = {b["box_id"]: b for b in next(s for s in f["sweeps"] if s["offset"] == -1)["boxes"]}
    assert boxes["2"]["review"]["status"] == "deleted" and boxes["1"]["review"]["final_bbox"] == [102, 401, 302, 521]
    assert (await client.post(f"{F}/sweeps/5/actions", json={"action": "KEEP", "box_id": "1"})).status_code == 422

    # Hoàn tác được như thao tác ở keyframe; sự kiện ghi cho báo cáo năng suất
    f = (await client.post(f"{F}/undo", json={})).json()
    assert next(s for s in f["sweeps"] if s["offset"] == -1)["boxes"][0]["review"]["status"] == "pending"
    assert sum(e["type"] == "sweep_action" for e in store.events()) == 4  # chỉ thao tác hợp lệ

    # Frame đã approve thì phải mở lại trước khi sửa sweep
    for oid in ("1", "2"):
        await client.post(f"{F}/actions", json={"action": "KEEP", "object_id": oid})
    assert (await client.post(f"{F}/approve", json={})).status_code == 200
    assert (await client.post(base, json={"action": "KEEP", "box_id": "1"})).status_code == 422


@pytest.mark.asyncio
async def test_sweep_edit_proposes_recovered_box(client, sweep_frame):
    # Vật detector sót ở keyframe nhưng người vẽ ở t-1 và t+1 -> đề xuất RECOVERED_BY_TRACK ở keyframe
    for o in (-1, 1):
        f = (await client.post(f"{F}/sweeps/{o}/actions",
                               json={"action": "ADD_BOX", "bbox": [1200, 500, 1300, 600], "label": "car"})).json()  # fmt: skip
    rec = [o for o in f["objects"] if o["source"] == "track"]
    assert len(rec) == 1 and rec[0]["bbox"] == pytest.approx([1200, 500, 1300, 600])
    assert "RECOVERED_BY_TRACK" in [i["code"] for i in rec[0]["qa"]["issues"]]
    # Xoá một bên -> đề xuất còn chờ biến mất
    f = (await client.post(f"{F}/sweeps/1/actions", json={"action": "DELETE", "box_id": "h1"})).json()
    assert not [o for o in f["objects"] if o["source"] == "track"]
