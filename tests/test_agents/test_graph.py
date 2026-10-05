import numpy as np

from src.agents.graph import qa_agent
from src.models.schemas import Detection, LabelObject
from tests.conftest import INTRINSIC


def test_qa_agent_end_to_end(config):
    car = LabelObject(object_id="1", bbox=[500, 400, 700, 500], label="car", score=0.9)
    ghost = LabelObject(object_id="2", bbox=[1000, 200, 1100, 500], label="pedestrian", score=0.3)

    xs, ys = np.meshgrid(np.linspace(510, 690, 10), np.linspace(410, 490, 10))
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1)
    depth = np.full(len(uv), 20.0)

    def sweep(ts, extra=()):
        return {"timestamp": ts, "detections": [Detection(bbox=[502, 400, 702, 500], label="car", score=0.9), *extra]}

    # Box sweep người đã xác nhận (score 1.0): từ 04/10/2026 chỉ box loại này tạo đề xuất RECOVERED_BY_TRACK
    missed = Detection(bbox=[200, 420, 300, 480], label="car", score=1.0)
    sweeps = {-2: sweep(800), -1: sweep(900, [missed]), 1: sweep(1050, [missed]), 2: sweep(1150)}

    out = qa_agent.invoke(
        {
            "config": config,
            "image_size": (1600, 900),
            "intrinsic": INTRINSIC,
            "key_timestamp": 1000,
            "objects": [car, ghost],
            "sweeps": sweeps,
            "lidar_uv": uv,
            "lidar_depth": depth,
        }
    )

    reviewed = {o.object_id: o for o in out["reviewed"]}
    assert set(reviewed) == {"1", "2", "r1"}

    assert reviewed["1"].qa.level == "low"
    assert reviewed["1"].qa.issues == []
    assert reviewed["1"].track["-1"] == [502, 400, 702, 500]

    ghost_codes = {i.code for i in reviewed["2"].qa.issues}
    assert {"LOW_CONFIDENCE", "NO_LIDAR_SUPPORT", "FLICKER"} <= ghost_codes
    assert reviewed["2"].qa.level == "high"

    rec = reviewed["r1"]
    assert rec.source == "track"
    assert "RECOVERED_BY_TRACK" in {i.code for i in rec.qa.issues}
    assert rec.qa.level != "low"

    # Sắp theo risk giảm dần và frame_risk = max
    risks = [o.qa.risk for o in out["reviewed"]]
    assert risks == sorted(risks, reverse=True)
    assert out["frame_risk"] == risks[0]


def test_qa_agent_without_lidar_or_sweeps(config):
    obj = LabelObject(object_id="1", bbox=[500, 400, 700, 500], label="car", score=0.95)
    out = qa_agent.invoke(
        {
            "config": config,
            "image_size": (1600, 900),
            "intrinsic": INTRINSIC,
            "key_timestamp": 0,
            "objects": [obj],
            "sweeps": {},
            "lidar_uv": None,
            "lidar_depth": None,
        }
    )
    (o,) = out["reviewed"]
    assert o.qa.issues == []
    assert o.qa.lidar == {"available": False}
    assert o.qa.level == "low"
