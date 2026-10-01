"""Thí nghiệm "keyframe hoàn hảo" cho lan truyền 3D: nhãn gốc ở keyframe 0, 5, 10… của mỗi scene đóng vai box người đã
duyệt, lan truyền qua các keyframe sau bằng dự đoán của mô hình, so với nhãn gốc cùng `instance_token`.

Mỗi box lan truyền ra ở keyframe đích:
    đúng    : tâm cách nhãn gốc của chính vật đó ≤ 2 m (ngưỡng TP rộng nhất của nuScenes detection)
    đổi ID  : sai, nhưng trùng nhãn gốc của vật khác (track nhảy sang vật khác)
    sai     : còn lại
Vật vẫn có mặt (có điểm LiDAR) mà không có box lan truyền: "mất". "Mô hình tự thấy" = có dự đoán cùng lớp ≤ 2 m (nhãn
máy người vẫn phải duyệt) — để so: lan truyền mang được bao nhiêu nhãn đã chốt so với việc duyệt lại từ đầu.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.models.qa_config import Propagation3DCfg
from src.services.propagation3d import MATCH_DIST, Det3D, Track3D, Tracker3D

TP_DIST = 2.0


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gt_global(data, token: str, category_map: dict) -> list[dict]:
    out = []
    for a in data._annotations_of_sample.get(token, []):
        label = category_map.get(data.category_of_instance[a["instance_token"]])
        if label is None or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
            continue
        out.append({"label": label, "center": np.asarray(a["translation"], float), "yaw": _yaw(a["rotation"]),
                    "size": list(a["size"]), "inst": a["instance_token"]})  # fmt: skip
    return out


def dets_global(boxes: list[dict], min_score: float) -> list[Det3D]:
    return [Det3D(np.asarray(b["translation"], float), _yaw(b["rotation"]), np.asarray(b.get("velocity", [0, 0])[:2], float),
                  b["detection_name"], b["detection_score"], i)
            for i, b in enumerate(boxes) if b["detection_score"] >= min_score]  # fmt: skip


def scene_keyframes(data, scene_name: str) -> list[tuple[str, int]]:
    sc = next(s for s in data.scene.values() if s["name"] == scene_name)
    out, tok = [], sc["first_sample_token"]
    while tok:
        out.append((tok, data.sample[tok]["timestamp"]))
        tok = data.sample[tok]["next"]
    return out


def evaluate(data, preds: dict[str, list[dict]], scenes: list[str], cfg: Propagation3DCfg, category_map: dict,
             min_score: float = 0.3, start_every: int = 5, max_frames: int = 10) -> dict:  # fmt: skip
    acc = defaultdict(float)
    per_hop: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    err = []
    for scene in scenes:
        kfs = scene_keyframes(data, scene)
        for start in range(0, max(1, len(kfs) - 1), start_every):
            tok0, _ = kfs[start]
            gts0 = gt_global(data, tok0, category_map)
            d0 = dets_global(preds.get(tok0, []), min_score)
            tracks = []
            for g in gts0:
                near = [d for d in d0 if d.label == g["label"]
                        and np.hypot(*(d.center[:2] - g["center"][:2])) <= MATCH_DIST.get(g["label"], 2.0)]  # fmt: skip
                vel = min(near, key=lambda d: np.hypot(*(d.center[:2] - g["center"][:2]))).vel if near else np.zeros(2)
                tracks.append(Track3D(g["inst"], g["label"], g["size"], g["center"].copy(), g["yaw"], vel.copy(), "keep"))
            if not tracks:
                continue
            acc["starts"] += 1
            acc["tracks"] += len(tracks)
            tracker = Tracker3D(tracks, cfg)
            started = {t.track_id for t in tracks}
            for hop in range(1, max_frames + 1):
                if start + hop >= len(kfs):
                    break
                tok, ts = kfs[start + hop]
                dt = (ts - kfs[start + hop - 1][1]) / 1e6
                dets = dets_global(preds.get(tok, []), min_score)
                tracker.step(dets, dt)
                gts = gt_global(data, tok, category_map)
                by_inst = {g["inst"]: g for g in gts}
                s = per_hop[hop]
                out_ids = set()
                for t in tracker.tracks:
                    if not t.alive or t.match is None:
                        continue
                    out_ids.add(t.track_id)
                    s["outputs"] += 1
                    own = by_inst.get(t.track_id)
                    d_own = float(np.hypot(*(t.center[:2] - own["center"][:2]))) if own else math.inf
                    if d_own <= TP_DIST:
                        s["correct"] += 1
                        err.append(d_own)
                    elif any(np.hypot(*(t.center[:2] - g["center"][:2])) <= TP_DIST for i, g in by_inst.items()
                             if i != t.track_id):  # fmt: skip
                        s["id_switch"] += 1
                    else:
                        s["wrong"] += 1
                for inst, g in by_inst.items():
                    if inst not in started:
                        continue
                    s["present"] += 1
                    if inst not in out_ids:
                        s["lost"] += 1
                    if any(d.label == g["label"] and np.hypot(*(d.center[:2] - g["center"][:2])) <= TP_DIST for d in dets):
                        s["model_sees"] += 1
                if not tracker.alive:
                    break
    tot = defaultdict(float)
    for s in per_hop.values():
        for k, v in s.items():
            tot[k] += v
    n = tot["outputs"]
    return {
        "starts": int(acc["starts"]), "tracks_started": int(acc["tracks"]), "outputs": int(n),
        "correct": int(tot["correct"]), "id_switch": int(tot["id_switch"]), "wrong": int(tot["wrong"]),
        "lost": int(tot["lost"]), "present": int(tot["present"]), "model_sees": int(tot["model_sees"]),
        "correct_rate": round(tot["correct"] / n, 4) if n else None,
        "coverage": round(tot["correct"] / tot["present"], 4) if tot["present"] else None,
        "mean_center_err_m": round(float(np.mean(err)), 3) if err else None,
        "per_hop": {h: {k: int(v) for k, v in s.items()} for h, s in sorted(per_hop.items())},
    }  # fmt: skip
