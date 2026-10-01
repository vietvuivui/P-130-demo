"""FR-19 báo cáo CSV và FR-27 năng suất theo người / phiên, throughput auto-label."""

import csv
import io
from datetime import UTC, datetime, timedelta

import pytest

from src.services import productivity
from tests.conftest import make_frame

F = "/api/v1/frames/scene-0001_000"


def _approved(fid, who, at, secs, run="run2d-20260929T010000-cuda", al=0.5):
    f = make_frame(fid)
    for o in f.objects:
        o.review.status = "approved"
    f.status, f.approved_by, f.approved_at, f.review_time_s = "approved", who, at.isoformat(timespec="seconds"), secs
    f.autolabel_s, f.autolabel_run = al, run
    return f


def test_reviewer_sessions_and_inference():
    t = datetime(2026, 9, 29, 8, 0, tzinfo=UTC)
    frames = [
        _approved("scene-0001_000", "an", t, 60),
        _approved("scene-0001_001", "an", t + timedelta(minutes=5), 120),
        _approved("scene-0001_002", "an", t + timedelta(hours=2), 60, run="video-20260929T090000-cpu", al=2.0),
        _approved("scene-0001_003", "binh", t, 30),
    ]
    stats = {r["reviewer"]: r for r in productivity.reviewer_stats(frames, [], [], "2d")}
    an = stats["an"]
    assert an["frames_approved"] == 3 and an["review_time_s"] == 240 and an["frames_per_hour"] == 45.0
    assert [s["frames"] for s in an["sessions"]] == [2, 1]  # nghỉ 2 giờ -> phiên mới
    assert an["sessions"][0]["frames_per_hour"] == 40.0
    inf = {r["run"]: r for r in productivity.inference_stats(frames)}
    assert inf["run2d-20260929T010000-cuda"]["frames"] == 3 and inf["run2d-20260929T010000-cuda"]["device"] == "cuda"
    assert inf["video-20260929T090000-cpu"]["frames_per_hour"] == 1800.0


@pytest.mark.asyncio
async def test_metrics_and_csv(client, store):
    for oid in ("1", "2"):
        await client.post(f"{F}/actions", json={"action": "KEEP", "object_id": oid, "reviewer": "an"})
    await client.post(f"{F}/actions", json={"action": "DELETE", "object_id": "3", "reviewer": "an"})
    await client.post(f"{F}/approve", json={"reviewer": "an", "review_time_s": 90})
    m = (await client.get("/api/v1/metrics")).json()
    r = m["productivity"]["reviewers"][0]
    assert (
        r["reviewer"] == "an" and r["frames_approved"] == 1 and r["objects_handled"] == 3 and r["frames_per_hour"] == 40
    )
    res = await client.get("/api/v1/report.csv", params={"kind": "frames"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(res.text.lstrip("﻿"))))
    assert rows[0]["frame_id"] == "scene-0001_000" and rows[0]["fixed"] == "1" and rows[0]["approved_by"] == "an"
    summary = (await client.get("/api/v1/report.csv", params={"kind": "summary"})).text
    assert "m4_correction_rate" in summary and "productivity.reviewers[0].frames_per_hour" in summary
