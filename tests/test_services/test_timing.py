"""Đo thời gian từng bước (timing.py) và API /timing."""

import pytest

from src.services import timing


def test_stopwatch_and_summary():
    sw = timing.Stopwatch()
    with sw("detect"):
        pass
    sw.add("detect", 0.5)
    sw.add("n_model_images", 5)
    assert sw.result()["detect"] >= 0.5
    rows = timing.summarize([{"detect": 1.0, "qa": 0.1, "n_model_images": 5}, {"detect": 3.0, "lidar": 0.4}])
    assert [r["stage"] for r in rows] == ["detect", "lidar", "qa"]  # xếp theo tổng thời gian
    assert rows[0]["mean_s"] == 2.0 and rows[0]["share"] == pytest.approx(4 / 4.5, abs=1e-3)
    timing.record("load_model", 2.0)
    assert timing.snapshot()["load_model"]["max_s"] >= 2.0


@pytest.mark.asyncio
async def test_timing_endpoint(client, store):
    f = store.load_frame("scene-0001_000")
    f.autolabel_s, f.autolabel_timing = 1.2, {"lidar": 0.1, "detect": 1.0, "qa": 0.1, "n_model_images": 5}
    store.save_frame(f)
    r = (await client.get("/api/v1/timing")).json()
    assert r["frames2d"] == 1 and r["rows2d"][0]["stage"] == "detect" and r["model_images_per_frame"] == 5
    assert r["mean_total2d"] == 1.2 and r["steps"] == []
