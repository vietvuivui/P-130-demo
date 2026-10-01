"""Bộ đo tracking (HOTA / MOTA / IDF1) trong repo: trường hợp biên và đối chiếu với TrackEval chính thức nếu có."""

import numpy as np
import pytest

from src.services import trackeval as te


def _seq(n=10, switch_at=None, miss=(), extra=()):
    """Một xe chạy sang phải; tracker giữ id 'a', đổi sang 'b' từ frame switch_at; bỏ frame trong miss; thêm box lạ."""
    frames = []
    for t in range(n):
        box = [100.0 + 10 * t, 100.0, 200.0 + 10 * t, 160.0]
        gt = [te.Obs("g1", "car", box)]
        pr = []
        if t not in miss:
            pr.append(te.Obs("b" if switch_at is not None and t >= switch_at else "a", "car", list(box)))
        if t in extra:
            pr.append(te.Obs("x", "car", [900.0, 100.0, 1000.0, 160.0]))
        frames.append((f"f{t}", gt, pr))
    return frames


def test_perfect_tracker():
    r = te.evaluate_tracks(_seq(), te.iou_sim)
    assert r["HOTA"] > 0.95 and r["MOTA"] == 1.0 and r["IDF1"] == 1.0 and r["IDSW"] == 0


def test_id_switch_and_misses_are_penalised():
    r = te.evaluate_tracks(_seq(switch_at=5), te.iou_sim)
    assert r["IDSW"] == 1 and r["MOTA"] == pytest.approx(0.9) and r["IDF1"] == pytest.approx(0.5) and r["AssA"] < 0.6
    r2 = te.evaluate_tracks(_seq(miss=(2, 3)), te.iou_sim)
    assert r2["FN"] == 2 and r2["MOTA"] == pytest.approx(0.8) and r2["DetA"] < r["DetA"] + 0.3
    r3 = te.evaluate_tracks(_seq(extra=(1,)), te.iou_sim)
    assert r3["FP"] == 1 and r3["MOTA"] == pytest.approx(0.9)


def test_class_and_ignore():
    frames = _seq(n=3)
    frames[0][2][0].label = "truck"  # khác lớp -> không khớp
    r = te.evaluate_tracks(frames, te.iou_sim)
    assert r["FP"] == 1 and r["FN"] == 1
    frames = _seq(n=3)
    frames[1][1][0].ignore = True  # GT khó thấy: bỏ cả GT lẫn dự đoán khớp nó
    r = te.evaluate_tracks(frames, te.iou_sim)
    assert r["num_gt_dets"] == 2 and r["num_pred_dets"] == 2 and r["MOTA"] == 1.0


def test_center_similarity_3d():
    frames = [
        (f"f{t}", [te.Obs("g", "car", [10.0 + t, 5.0])], [te.Obs("p", "car", [10.0 + t + 0.5, 5.0])]) for t in range(4)
    ]
    r = te.evaluate_tracks(frames, te.center_sim(2.0))
    assert r["MOTA"] == 1.0 and 0.5 < r["HOTA"] < 1.0
    frames[2][2][0].geom = [100.0, 100.0]
    r = te.evaluate_tracks(frames, te.center_sim(2.0))
    assert r["FP"] == 1 and r["FN"] == 1


def _random_seq(rng, n_frames=30, n_obj=6):
    """Chuỗi ngẫu nhiên có lỗi đủ loại để đối chiếu công thức."""
    frames = []
    for t in range(n_frames):
        gt, pr = [], []
        for k in range(n_obj):
            x = 100.0 + 150 * k + 3 * t
            box = [x, 100.0, x + 100, 180.0]
            gt.append(te.Obs(f"g{k}", "car", box))
            if rng.random() < 0.85:
                tid = f"p{k}" if (t < 15 or k % 2 == 0) else f"q{k}"
                jitter = rng.normal(0, 6, 4)
                pr.append(
                    te.Obs(tid, "car", [box[0] + jitter[0], box[1] + jitter[1], box[2] + jitter[2], box[3] + jitter[3]])
                )
        if rng.random() < 0.3:
            pr.append(te.Obs("fp", "car", [1200.0, 300.0, 1300.0, 380.0]))
        frames.append((f"f{t}", gt, pr))
    return frames


def test_matches_official_trackeval(tmp_path):
    trackeval = pytest.importorskip("trackeval")  # noqa: F841
    frames = _random_seq(np.random.default_rng(1))
    ours = te.evaluate_tracks(frames, te.iou_sim)
    te.export_mot({"seq01": frames}, tmp_path, "t")
    official = te.run_trackeval(tmp_path, "t")
    assert official is not None
    for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        assert ours[k] == pytest.approx(official[k], abs=1e-3), (k, ours[k], official[k])
    assert ours["IDSW"] == official["IDSW"]
