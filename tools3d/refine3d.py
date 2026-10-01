"""Tinh chỉnh kết quả 3D không cần train: gộp nhiều mô hình (ensemble có hướng) và tinh chỉnh theo track.

Làm việc trên file kết quả chuẩn nuScenes (hệ toàn cục), chỉ dùng numpy, để chạy được cả trong môi trường
MMDetection3D (tools3d/run3d.py refine) lẫn môi trường chính.

- fuse(): gom box của nhiều mô hình theo lớp và khoảng cách tâm, lấy trung bình có trọng số điểm (hướng được căn
  cùng chiều trước khi lấy trung bình). Điểm = trung bình điểm x (số mô hình đồng ý / số mô hình), như WBF.
- refine_tracks(): nối box qua các keyframe thành track (khớp tham lam theo vị trí dự đoán bằng vận tốc, ngưỡng theo
  lớp của CenterPoint), rồi: kích thước = trung vị theo track, vận tốc = sai phân trung tâm của vị trí trong track,
  hướng lật 180° cho khớp đa số track, điểm = trộn điểm box với điểm trung bình track, nội suy keyframe bị hụt.

Tham số lấy theo mặc định trong CenterPoint / WBF, không dò theo nhãn gốc.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# Bán kính gom box giữa các mô hình (m, khoảng cách tâm BEV): ~nửa kích thước điển hình của lớp
FUSE_RADIUS = {
    "car": 1.0, "truck": 1.5, "bus": 2.0, "trailer": 2.0, "construction_vehicle": 1.5, "pedestrian": 0.5,
    "motorcycle": 0.8, "bicycle": 0.8, "traffic_cone": 0.4, "barrier": 0.8,
}  # fmt: skip
# Sai số vị trí tối đa khi nối track giữa hai keyframe (m), theo tracker của CenterPoint (NUSCENE_CLS_VELOCITY_ERROR)
TRACK_DIST = {
    "car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0, "construction_vehicle": 3.0, "pedestrian": 1.0,
    "motorcycle": 13.0, "bicycle": 3.0, "traffic_cone": 1.0, "barrier": 1.0,
}  # fmt: skip
STATIC = {"traffic_cone", "barrier"}  # vật đứng yên: vận tốc 0


def yaw_of(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_of(yaw: float) -> list[float]:
    return [round(math.cos(yaw / 2), 6), 0.0, 0.0, round(math.sin(yaw / 2), 6)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _align(yaw: float, ref: float) -> float:
    """yaw hoặc yaw + pi, cái nào gần ref hơn (box 3D hay bị đoán ngược đầu)."""
    return yaw if abs(_wrap(yaw - ref)) <= math.pi / 2 else _wrap(yaw + math.pi)


def _mean_yaw(yaws, weights) -> float:
    return math.atan2(
        sum(w * math.sin(y) for y, w in zip(yaws, weights)), sum(w * math.cos(y) for y, w in zip(yaws, weights))
    )


# ---------------------------------------------------------------- ensemble
def fuse(results_by_model: dict[str, dict[str, list[dict]]], weights: dict[str, float] | None = None,
         min_score: float = 0.0) -> dict[str, list[dict]]:  # fmt: skip
    """Gộp kết quả nhiều mô hình. results_by_model[model][sample_token] = list box nuScenes."""
    models = list(results_by_model)
    weights = weights or {m: 1.0 for m in models}
    total_w = sum(weights.values())
    tokens = set().union(*[set(r) for r in results_by_model.values()])
    out = {}
    for tok in tokens:
        by_cls = defaultdict(list)
        for m in models:
            for b in results_by_model[m].get(tok, []):
                if b["detection_score"] >= min_score:
                    by_cls[b["detection_name"]].append((m, b))
        fused = []
        for cls, items in by_cls.items():
            items.sort(key=lambda mb: -mb[1]["detection_score"])
            xy = np.array([mb[1]["translation"][:2] for mb in items])
            used = np.zeros(len(items), bool)
            radius = FUSE_RADIUS.get(cls, 1.0)
            for i in range(len(items)):
                if used[i]:
                    continue
                d = np.hypot(*(xy - xy[i]).T)
                cand = np.where(~used & (d <= radius))[0]
                members, seen = [], set()
                for j in cand:  # mỗi mô hình góp tối đa một box (box điểm cao nhất, vì đã sắp xếp)
                    m = items[j][0]
                    if m in seen:
                        continue
                    seen.add(m)
                    members.append(items[j])
                    used[j] = True
                used[i] = True
                fused.append(_merge(cls, members, weights, total_w))
        out[tok] = fused
    return out


def _merge(cls: str, members: list, weights: dict, total_w: float) -> dict:
    lead = members[0][1]
    w = np.array([weights[m] * b["detection_score"] for m, b in members])
    ref = yaw_of(lead["rotation"])
    yaws = [_align(yaw_of(b["rotation"]), ref) for _, b in members]
    tr = np.average([b["translation"] for _, b in members], axis=0, weights=w)
    size = np.average([b["size"] for _, b in members], axis=0, weights=w)
    vel = np.average([b.get("velocity", [0, 0])[:2] for _, b in members], axis=0, weights=w)
    agree = sum(weights[m] for m, _ in members) / total_w
    score = float(np.average([b["detection_score"] for _, b in members], weights=[weights[m] for m, _ in members]))
    return {
        "sample_token": lead["sample_token"], "translation": tr.round(3).tolist(), "size": size.round(3).tolist(),
        "rotation": quat_of(_mean_yaw(yaws, w)), "velocity": vel.round(3).tolist(), "detection_name": cls,
        "detection_score": round(score * agree, 4), "attribute_name": lead.get("attribute_name", ""),
    }  # fmt: skip


# ---------------------------------------------------------------- tinh chỉnh theo track
def build_tracks(results: dict[str, list[dict]], scene_samples: list[tuple[str, float]], min_score: float = 0.1,
                 max_gap: int = 3):  # fmt: skip
    """Nối box qua các keyframe của một scene. scene_samples: [(sample_token, timestamp_s)] theo thời gian.

    Khớp tham lam theo điểm giảm dần, riêng từng lớp: vị trí dự đoán = vị trí cuối + vận tốc x thời gian, ngưỡng theo
    lớp (TRACK_DIST, nới theo căn số keyframe bị hụt). Track không được nối chờ tối đa max_gap keyframe.
    Trả về list track, mỗi track là list (chỉ số keyframe, box)."""
    tracks: list[list[tuple[int, dict]]] = []
    alive: dict[str, list[int]] = defaultdict(list)  # lớp -> chỉ số track còn chờ nối
    for k, (tok, t) in enumerate(scene_samples):
        by_cls = defaultdict(list)
        for b in results.get(tok, []):
            if b["detection_score"] >= min_score:
                by_cls[b["detection_name"]].append(b)
        for cls in set(by_cls) | set(alive):
            dets = sorted(by_cls.get(cls, []), key=lambda b: -b["detection_score"])
            cand = [ti for ti in alive[cls] if k - tracks[ti][-1][0] <= max_gap]
            matched: set[int] = set()
            if dets and cand:
                pred = np.empty((len(cand), 2))
                gate = np.empty(len(cand))
                for j, ti in enumerate(cand):
                    kp, pb = tracks[ti][-1]
                    dt = t - scene_samples[kp][1]
                    v = pb.get("velocity", [0.0, 0.0])
                    pred[j] = (pb["translation"][0] + v[0] * dt, pb["translation"][1] + v[1] * dt)
                    gate[j] = TRACK_DIST.get(cls, 2.0) * max(1, k - kp) ** 0.5
                xy = np.array([b["translation"][:2] for b in dets])
                dist = np.hypot(xy[:, None, 0] - pred[None, :, 0], xy[:, None, 1] - pred[None, :, 1])
                dist[dist > gate[None, :]] = np.inf
                assigned = {}
                for i in range(len(dets)):  # tham lam theo điểm giảm dần
                    row = dist[i].copy()
                    if matched:
                        row[list(matched)] = np.inf
                    j = int(np.argmin(row))
                    if np.isfinite(row[j]):
                        matched.add(j)
                        assigned[i] = cand[j]
            else:
                assigned = {}
            new_alive = []
            for i, b in enumerate(dets):
                if i in assigned:
                    tracks[assigned[i]].append((k, b))
                    new_alive.append(assigned[i])
                else:
                    tracks.append([(k, b)])
                    new_alive.append(len(tracks) - 1)
            # track chưa được nối vẫn chờ (vật bị che / detector sót)
            waiting = [ti for j, ti in enumerate(cand) if j not in matched]
            alive[cls] = new_alive + waiting
    return tracks


def refine_tracks(results: dict[str, list[dict]], scenes: dict[str, list[tuple[str, float]]], track_min_score: float = 0.05,
                  blend: float = 0.5, fill_gaps: bool = True, fix_size: bool = True, fix_velocity: bool = True,
                  fix_heading: bool = True, rescore: bool = True) -> dict[str, list[dict]]:  # fmt: skip
    """scenes[scene_token] = [(sample_token, timestamp_s)] theo thời gian. Box dưới track_min_score giữ nguyên.

    track_min_score 0.05 (trước là 0.1), chọn trên dev trong {0.02, 0.03, 0.05, 0.1, 0.2}: ensemble 4 mô hình dev mAP
    0.644 -> 0.654, test 24 scene 0.666 -> 0.671 (eval/results/lidar2d.md)."""
    out = {tok: [b for b in bs if b["detection_score"] < track_min_score] for tok, bs in results.items()}
    for si, samples in enumerate(scenes.values()):
        times = [t for _, t in samples]
        for ti, tr in enumerate(build_tracks(results, samples, track_min_score)):
            boxes = [dict(b, tracking_id=f"{si}-{ti}") for _, b in tr]
            ks = [k for k, _ in tr]
            scores = np.array([b["detection_score"] for b in boxes])
            cls = boxes[0]["detection_name"]
            if fix_size and len(boxes) >= 2:
                size = np.median([b["size"] for b in boxes], axis=0).round(3).tolist()
                for b in boxes:
                    b["size"] = size
            if fix_heading and len(boxes) >= 3:
                # hướng đa số (trung bình theo điểm, trục không phân biệt đầu/đuôi), rồi lật box ngược đầu
                yaws = [yaw_of(b["rotation"]) for b in boxes]
                ax = 0.5 * math.atan2(sum(s * math.sin(2 * y) for y, s in zip(yaws, scores)),
                                      sum(s * math.cos(2 * y) for y, s in zip(yaws, scores)))  # fmt: skip
                votes = sum(s if abs(_wrap(y - ax)) <= math.pi / 2 else -s for y, s in zip(yaws, scores))
                ref = ax if votes >= 0 else _wrap(ax + math.pi)
                for b, y in zip(boxes, yaws):
                    b["rotation"] = quat_of(_align(y, ref))
            if fix_velocity:
                for i, b in enumerate(boxes):
                    if cls in STATIC:
                        b["velocity"] = [0.0, 0.0]
                        continue
                    lo, hi = max(i - 1, 0), min(i + 1, len(boxes) - 1)
                    # chỉ dùng hai keyframe liền kề (không qua chỗ bị hụt) để sai phân
                    if hi == lo or ks[hi] - ks[lo] > 2:
                        continue
                    dt = times[ks[hi]] - times[ks[lo]]
                    if dt <= 0:
                        continue
                    v = [(boxes[hi]["translation"][j] - boxes[lo]["translation"][j]) / dt for j in range(2)]
                    vm = b.get("velocity", [0.0, 0.0])[:2]
                    # lệch quá xa vận tốc mô hình -> nhiều khả năng nối nhầm track: giữ vận tốc mô hình
                    if math.hypot(v[0] - vm[0], v[1] - vm[1]) > max(2.0, 0.5 * math.hypot(*vm)):
                        continue
                    b["velocity"] = [round(v[0], 3), round(v[1], 3)]
            if rescore:
                track_score = float(scores.mean())
                for b in boxes:
                    b["detection_score"] = round((1 - blend) * b["detection_score"] + blend * track_score, 4)
            for k, b in zip(ks, boxes):
                out.setdefault(samples[k][0], []).append(b)
            if fill_gaps:
                for (k0, b0), (k1, b1) in zip(zip(ks, boxes), zip(ks[1:], boxes[1:])):
                    for k in range(k0 + 1, k1):  # keyframe bị hụt giữa hai lần thấy
                        a = (times[k] - times[k0]) / max(times[k1] - times[k0], 1e-6)
                        y0 = yaw_of(b0["rotation"])
                        y1 = _align(yaw_of(b1["rotation"]), y0)
                        out.setdefault(samples[k][0], []).append({
                            "sample_token": samples[k][0],
                            "translation": [round(b0["translation"][i] * (1 - a) + b1["translation"][i] * a, 3) for i in range(3)],
                            "size": b0["size"], "rotation": quat_of(_wrap(y0 + a * _wrap(y1 - y0))),
                            "velocity": [round(b0["velocity"][i] * (1 - a) + b1["velocity"][i] * a, 3) for i in range(2)],
                            "detection_name": cls, "attribute_name": b0.get("attribute_name", ""),
                            "tracking_id": b0["tracking_id"],
                            "detection_score": round(0.5 * min(b0["detection_score"], b1["detection_score"]), 4),
                        })  # fmt: skip
    return out


def scenes_from_nusc(nusc, sample_tokens) -> dict[str, list[tuple[str, float]]]:
    """Nhóm các sample theo scene, sắp theo thời gian (timestamp µs -> s)."""
    by_scene = defaultdict(list)
    for tok in sample_tokens:
        s = nusc.get("sample", tok)
        by_scene[s["scene_token"]].append((tok, s["timestamp"] / 1e6))
    return {k: sorted(v, key=lambda x: x[1]) for k, v in by_scene.items()}


def cap_per_sample(results: dict[str, list[dict]], n: int = 500) -> dict[str, list[dict]]:
    """Bộ chấm nuScenes chỉ nhận tối đa 500 box / sample."""
    return {t: sorted(bs, key=lambda b: -b["detection_score"])[:n] for t, bs in results.items()}
