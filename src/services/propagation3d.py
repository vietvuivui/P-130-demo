"""Lan truyền box 3D đã duyệt sang các keyframe sau (idea 1 cho 3D).

Người approve một keyframe 3D; mọi quyết định (giữ, sửa box, đổi lớp, vẽ thêm, xoá) được mang sang các keyframe sau
còn "auto". Box được đưa về hệ toàn cục (bù chuyển động của xe tự lái bằng ego pose), dịch theo vận tốc của vật (mô
hình LiDAR dự đoán sẵn vận tốc, CenterPoint), rồi khớp với dự đoán của mô hình ở keyframe sau theo khoảng cách tâm
BEV — ngưỡng theo lớp của tracker CenterPoint. Lớp và kích thước khoá theo người; tâm / hướng lấy từ dự đoán ở chính
frame đó. Không khớp: track chạy tiếp theo vận tốc (không ghi ra), mất quá `max_misses` keyframe thì dừng.

    track (hệ toàn cục) --dự đoán c + v·dt--> khớp dự đoán mô hình ở K+1 --> object "propagated" (lớp / size người)

Nhãn lan truyền vẫn "pending"; khớp chặt và mô hình cùng lớp thì mức rủi ro low (duyệt theo lô), không thì medium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.models.qa_config import Propagation3DCfg
from src.models.schemas import ReviewState
from src.models.schemas3d import Box3D, Frame3DRecord, Object3D, Propagation3DInfo, Verify3D

# Sai số vị trí tối đa khi nối box giữa hai keyframe (m), theo tracker CenterPoint (NUSCENE_CLS_VELOCITY_ERROR)
MATCH_DIST = {
    "car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0, "construction_vehicle": 3.0, "pedestrian": 1.0,
    "motorcycle": 13.0, "bicycle": 3.0, "traffic_cone": 1.0, "barrier": 1.0,
}  # fmt: skip
STATIC = {"traffic_cone", "barrier"}  # vật đứng yên: không dịch theo vận tốc
REVIEWER = "propagation"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_global(box: Box3D, g_from_l: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    g = np.asarray(g_from_l, float)
    c = g[:3, :3] @ np.asarray(box.center, float) + g[:3, 3]
    yaw = _wrap(box.yaw + math.atan2(g[1, 0], g[0, 0]))
    v = g[:3, :3] @ np.array([*box.velocity[:2], 0.0])
    return c, yaw, v[:2]


def to_lidar(center: np.ndarray, yaw: float, vel: np.ndarray, size: list[float], g_from_l: np.ndarray) -> Box3D:
    li = np.linalg.inv(np.asarray(g_from_l, float))
    c = li[:3, :3] @ center + li[:3, 3]
    v = li[:3, :3] @ np.array([*vel[:2], 0.0])
    return Box3D(center=[round(float(x), 3) for x in c], size=[round(float(x), 3) for x in size],
                 yaw=round(_wrap(yaw + math.atan2(li[1, 0], li[0, 0])), 4),
                 velocity=[round(float(v[0]), 3), round(float(v[1]), 3)])  # fmt: skip


@dataclass
class Det3D:
    """Dự đoán của mô hình ở một keyframe, hệ toàn cục."""

    center: np.ndarray
    yaw: float
    vel: np.ndarray
    label: str
    score: float
    ref: object = None  # object_id (sản phẩm) hoặc chỉ số (đánh giá)


@dataclass
class Track3D:
    track_id: str
    label: str
    size: list[float]
    center: np.ndarray
    yaw: float
    vel: np.ndarray
    kind: str  # keep | suppress
    keyframe_id: str = ""
    keyframe_object_id: str = ""
    misses: int = 0
    hops: int = 0
    alive: bool = True
    match: Det3D | None = None
    dist: float | None = None
    extra: dict = field(default_factory=dict)
    # OC-SORT: quan sát thật gần nhất, thời gian từ đó tới giờ, vận tốc theo quan sát, hướng đi đã quan sát
    last_obs: np.ndarray | None = None
    since_obs: float = 0.0
    obs_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    obs_dir: np.ndarray | None = None
    recovered: int = 0


class Tracker3D:
    def __init__(self, tracks: list[Track3D], cfg: Propagation3DCfg):
        self.tracks = tracks
        self.cfg = cfg
        for t in tracks:  # box người chốt ở keyframe là quan sát đầu tiên (cho OCR)
            if t.last_obs is None:
                t.last_obs = t.center.copy()

    @property
    def alive(self) -> list[Track3D]:
        return [t for t in self.tracks if t.alive]

    def predict(self, t: Track3D, dt: float) -> np.ndarray:
        if self.cfg.motion == "velocity" and t.label not in STATIC:
            return t.center + np.array([t.vel[0] * dt, t.vel[1] * dt, 0.0])
        return t.center.copy()

    def step(self, dets: list[Det3D], dt: float) -> None:
        tracks = self.alive
        cfg = self.cfg
        preds = [self.predict(t, dt) for t in tracks]
        pairs: dict[int, tuple[int, float]] = {}
        used: set[int] = set()
        for kind in ("keep", "suppress"):
            cand = []
            for i, (t, p) in enumerate(zip(tracks, preds, strict=True)):
                if t.kind != kind:
                    continue
                thr = MATCH_DIST.get(t.label, 2.0) * cfg.dist_scale
                for j, d in enumerate(dets):
                    if kind == "suppress" and d.label != t.label:
                        continue  # chỉ tự xoá box cùng lớp với box người đã xoá
                    dist = float(np.hypot(*(d.center[:2] - p[:2])))
                    if dist <= thr:
                        cand.append((dist - self._momentum(t, d) * thr, dist, i, j))
            for _, dist, i, j in sorted(cand):
                if i in pairs or j in used:
                    continue
                pairs[i] = (j, dist)
                used.add(j)
        if cfg.oc_recover:
            # OCR: track đang mất ghép với dự đoán còn thừa theo tâm quan sát cuối dời theo vận tốc quan sát
            cand = []
            for i, t in enumerate(tracks):
                if i in pairs or t.kind != "keep" or t.misses == 0 or t.last_obs is None:
                    continue
                gap = t.since_obs + dt
                anchor = t.last_obs[:2] + (t.obs_vel * gap if t.label not in STATIC else 0.0)
                thr = MATCH_DIST.get(t.label, 2.0) * cfg.dist_scale * cfg.oc_recover_scale
                for j, d in enumerate(dets):
                    if j in used or d.label != t.label:
                        continue
                    dist = float(np.hypot(*(d.center[:2] - anchor)))
                    if dist <= thr:
                        cand.append((dist, i, j))
            for dist, i, j in sorted(cand):
                if i in pairs or j in used:
                    continue
                pairs[i] = (j, dist)
                used.add(j)
                tracks[i].recovered += 1
        limit = cfg.oc_max_lost if cfg.oc_recover else cfg.max_misses
        for i, t in enumerate(tracks):
            t.hops += 1
            if i in pairs:
                j, dist = pairs[i]
                d = dets[j]
                gap = t.since_obs + dt
                if cfg.oc_reupdate and t.misses > 0 and t.last_obs is not None:
                    moved = (d.center[:2] - t.last_obs[:2]) / max(gap, 1e-3)  # ORU: quỹ đạo ảo quan sát -> quan sát
                else:
                    moved = (d.center[:2] - t.center[:2]) / max(dt, 1e-3)
                t.vel = d.vel if np.hypot(*d.vel) > 0.05 or t.label in STATIC else moved
                if t.last_obs is not None:
                    step_v = (d.center[:2] - t.last_obs[:2]) / max(gap, 1e-3)
                    t.obs_vel = 0.5 * step_v + 0.5 * t.obs_vel if np.hypot(*t.obs_vel) > 0 else step_v
                    mv = d.center[:2] - t.last_obs[:2]
                    if np.hypot(*mv) > 0.3:
                        t.obs_dir = mv / np.hypot(*mv)
                t.last_obs, t.since_obs = d.center.copy(), 0.0
                t.center, t.yaw = d.center.copy(), d.yaw
                t.misses, t.match, t.dist = 0, d, dist
            else:
                t.center = preds[i]
                t.since_obs += dt
                t.misses, t.match, t.dist = t.misses + 1, None, None
                if t.misses > limit:
                    t.alive = False

    def _momentum(self, t: Track3D, d: Det3D) -> float:
        """OCM: oc_momentum x cos(hướng đi đã quan sát, hướng từ quan sát cuối tới dự đoán); 0 nếu chưa có hướng."""
        if self.cfg.oc_momentum <= 0 or t.obs_dir is None or t.last_obs is None:
            return 0.0
        w = d.center[:2] - t.last_obs[:2]
        n = float(np.hypot(*w))
        return 0.0 if n < 0.3 else self.cfg.oc_momentum * float(w @ t.obs_dir) / n


# ---------------------------------------------------------------- nối với frame 3D của sản phẩm


def frame_dets(frame: Frame3DRecord, objects: list[Object3D]) -> list[Det3D]:
    out = []
    for o in objects:
        if o.source != "model":
            continue
        c, yaw, v = to_global(o.box, frame.global_from_lidar)
        out.append(Det3D(c, yaw, v, o.label, o.score, o.object_id))
    return out


def init_tracks3d(keyframe: Frame3DRecord) -> list[Track3D]:
    """Track từ quyết định của người ở keyframe. Gán track_id cho object (sửa tại chỗ)."""
    tracks = []
    for o in keyframe.objects:
        r = o.review
        if r.status == "approved":
            kind, label = "keep", r.final_label or o.label
        elif r.status == "deleted" and o.source == "model":
            kind, label = "suppress", o.label
        else:
            continue
        o.track_id = o.track_id or f"{keyframe.frame_id}:{o.object_id}"
        c, yaw, v = to_global(o.box, keyframe.global_from_lidar)
        tracks.append(Track3D(o.track_id, label, list(o.box.size), c, yaw, v, kind, keyframe.frame_id, o.object_id))
    return tracks


def apply_tracks(frame: Frame3DRecord, tracks: list[Track3D], at: str) -> dict:
    """Ghi track đang khớp vào keyframe đích (dựng lại từ prelabel, nên lan truyền lại không cộng dồn)."""
    if frame.prelabel is None:
        frame.prelabel = [o.model_copy(deep=True) for o in frame.objects]
    objects = [o.model_copy(deep=True) for o in frame.prelabel]
    by_id = {o.object_id: o for o in objects}
    propagated = suppressed = 0
    for t in tracks:
        if not t.alive or t.match is None or t.match.ref not in by_id:
            continue
        o = by_id[t.match.ref]
        if t.kind == "suppress":
            o.review = ReviewState(status="deleted", action="PROPAGATED_DELETE", reviewer=REVIEWER, at=at)
            suppressed += 1
            continue
        same = o.label == t.label
        tight = t.dist is not None and t.dist <= 0.5 * MATCH_DIST.get(t.label, 2.0)
        level = "low" if same and tight else "medium"
        note = (f"Lan truyền từ {t.keyframe_id} (#{t.keyframe_object_id}), lệch {t.dist:.1f} m sau dự đoán chuyển động"
                + ("" if same else f"; mô hình ở frame này gọi là {o.label}"))  # fmt: skip
        verify = o.verify.model_copy(update={"level": level, "comment": note + (f". {o.verify.comment}" if o.verify.comment else "")}) \
            if o.verify else Verify3D(verdict="CHUA DU THONG TIN", level=level, comment=note)  # fmt: skip
        o.label = t.label
        o.box = o.box.model_copy(update={"size": [round(float(x), 3) for x in t.size]})  # kích thước người đã chốt
        o.source, o.track_id, o.verify = "propagated", t.track_id, verify
        o.propagation = Propagation3DInfo(keyframe_id=t.keyframe_id, keyframe_object_id=t.keyframe_object_id,
                                          distance_m=round(t.dist or 0.0, 2), hops=t.hops)  # fmt: skip
        propagated += 1
    frame.objects = objects
    frame.propagated_from, frame.propagated_at = tracks[0].keyframe_id if tracks else None, at
    frame.frame_risk = max(({"low": 0.2, "medium": 0.5, "high": 0.8}.get(o.verify.level, 0.5)
                            for o in objects if o.verify and o.review.status == "pending"), default=0.0)  # fmt: skip
    return {"propagated": propagated, "suppressed": suppressed}


def propagate3d_from(store, model: str, keyframe_id: str, cfg: Propagation3DCfg, timestamps=None,
                     max_frames: int | None = None) -> dict:  # fmt: skip
    """Lan truyền từ keyframe 3D đã approve sang các keyframe sau cùng scene còn "auto".

    timestamps(sample_token) -> µs (bảng nuScenes); None thì coi hai keyframe liên tiếp cách nhau 0.5 s."""
    from src.services.review import now_iso

    key = store.load_frame3d(model, keyframe_id)
    if key is None:
        raise ValueError(f"Không có frame 3D {keyframe_id}")
    if key.status != "approved":
        raise ValueError("Chỉ lan truyền từ frame đã approve")
    tracks = init_tracks3d(key)
    store.save_frame3d(key)
    resp = {"keyframe_id": keyframe_id, "tracks_started": len(tracks), "frames_updated": [], "objects_propagated": 0,
            "objects_suppressed": 0, "stop_reason": None}  # fmt: skip
    if not any(t.kind == "keep" for t in tracks):
        resp["stop_reason"] = "Keyframe không có box nào đã duyệt để lan truyền"
        return resp
    after = sorted((f for f in store.list_frames3d(model) if f.scene == key.scene and f.index > key.index),
                   key=lambda f: f.index)  # fmt: skip
    tracker = Tracker3D(tracks, cfg)
    prev, at = key, now_iso()
    max_frames = max_frames or cfg.max_keyframes
    for n, f in enumerate(after, start=1):
        if f.status != "auto":
            resp["stop_reason"] = f"Dừng trước {f.frame_id}: frame đã có người mở hoặc duyệt"
            break
        ts = (timestamps(f.sample_token) - timestamps(prev.sample_token)) / 1e6 if timestamps else None
        dt = ts if ts and ts > 0 else 0.5 * (f.index - prev.index)
        base = f.prelabel if f.prelabel is not None else f.objects
        tracker.step(frame_dets(f, base), dt)
        res = apply_tracks(f, tracker.tracks, at)
        store.save_frame3d(f)
        resp["frames_updated"].append(f.frame_id)
        resp["objects_propagated"] += res["propagated"]
        resp["objects_suppressed"] += res["suppressed"]
        prev = f
        if not any(t.kind == "keep" for t in tracker.alive):
            resp["stop_reason"] = "Mọi box đã dừng lan truyền (mất dấu hoặc ra khỏi vùng quét)"
            break
        if n >= max_frames:
            resp["stop_reason"] = f"Đã đi đủ {max_frames} keyframe"
            break
    else:
        resp["stop_reason"] = resp["stop_reason"] or "Hết scene"
    resp["tracks_alive"] = sum(t.kind == "keep" for t in tracker.alive)
    return resp
