"""Lan truyền nhãn trên cả scene: nối thuật toán (propagation.py) với dữ liệu và workspace.

- SequenceSource: nguồn ảnh theo thời gian + detection đã cache. Bản thật đọc nuScenes và cache
  detector; test dùng bản giả, nên phần này chạy được mà không cần GPU hay dataset.
- propagate_from(): lan truyền từ một keyframe đã approve, ghi vào các keyframe sau còn "auto".
- evaluate_propagation(): thí nghiệm "keyframe hoàn hảo" — lấy GT ở keyframe đầu scene làm quyết
  định của người, lan truyền, so với GT cùng instance_token ở các keyframe sau. Đo được chất lượng
  lan truyền trước khi có người gán nhãn thật.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import Detection, FrameRecord, LabelObject, PropagateResponse, PropagateSkip, ReviewState
from src.services.detectors import DetectorEnsemble
from src.services.flow import FlowProvider
from src.services.geometry import iou
from src.services.nuscenes_data import NuScenesMini, TimelineImage
from src.services.propagation import (
    PROPAGATION_REVIEWER,
    Track,
    Tracker,
    apply_propagation,
    init_tracks,
    lidar_arrays,
    track_confidence,
)
from src.services.review import now_iso
from src.services.store import WorkspaceStore
from src.services.sweep_review import effective_detections


class SequenceSource(Protocol):
    def timeline(self, scene: str, camera: str) -> list[TimelineImage]: ...

    def detections(self, sd_token: str) -> list[Detection] | None: ...


def _motion(source, cfg) -> FlowProvider | None:
    """Optical flow cho tracker (propagation.flow) nếu nguồn đọc được file ảnh."""
    image_file = getattr(source, "image_file", None)
    if cfg.flow == "off" or image_file is None:
        return None
    return FlowProvider(image_file, cfg.flow_scale)


class NuScenesSequenceSource:
    """Timeline từ bảng nuScenes, detection từ cache của DetectorEnsemble (không chạy model)."""

    def __init__(self, data: NuScenesMini, ensemble: DetectorEnsemble):
        self.data = data
        self.ensemble = ensemble
        self._timelines: dict[tuple[str, str], list[TimelineImage]] = {}

    def timeline(self, scene: str, camera: str) -> list[TimelineImage]:
        key = (scene, camera)
        if key not in self._timelines:
            self._timelines[key] = self.data.camera_timeline(scene, camera)
        return self._timelines[key]

    def detections(self, sd_token: str) -> list[Detection] | None:
        return self.ensemble.load_cached(sd_token)

    def image_file(self, path: str) -> Path:
        return Path(self.data.dataroot) / path


class PropagationError(ValueError):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


class WorkspaceSequenceSource:
    """Nguồn dùng cho API: video tải lên lấy timeline từ record trong workspace; scene nuScenes đọc bảng nuScenes
    (chỉ nạp khi cần, nên lan truyền trên video tải lên chạy được cả khi máy không có dataset)."""

    def __init__(self, store: WorkspaceStore, ensemble: DetectorEnsemble, nuscenes: Callable[[], NuScenesMini],
                 dataroot: str | Path | None = None):  # fmt: skip
        self.store = store
        self.ensemble = ensemble
        self.dataroot = dataroot
        self._nuscenes_factory = nuscenes
        self._nuscenes: NuScenesSequenceSource | None = None

    def timeline(self, scene: str, camera: str) -> list[TimelineImage]:
        video = self.store.load_video(scene)
        if video is not None:
            return [TimelineImage(**e.model_dump()) for e in video.timeline]
        if self._nuscenes is None:
            try:
                self._nuscenes = NuScenesSequenceSource(self._nuscenes_factory(), self.ensemble)
            except FileNotFoundError as e:
                raise PropagationError(
                    "DATA_UNAVAILABLE", f"Lan truyền trên scene nuScenes cần bảng nuScenes: {e}", 503
                ) from e
        return self._nuscenes.timeline(scene, camera)

    def detections(self, sd_token: str) -> list[Detection] | None:
        return self.ensemble.load_cached(sd_token)

    def image_file(self, path: str) -> Path:
        from src.services.pipeline import resolve_image

        return resolve_image(self.dataroot or ".", self.store.root, path)


def sweep_overrides(frames: list[FrameRecord]) -> dict[str, list[Detection]]:
    """Detection của các sweep người đã sửa (sd_token -> box sau khi sửa): thay cho cache detector khi tracking."""
    return {s.sd_token: effective_detections(s) for f in frames for s in f.sweeps if s.boxes is not None}


def _timeline_after(source: SequenceSource, keyframe: FrameRecord) -> list[TimelineImage]:
    timeline = source.timeline(keyframe.scene, keyframe.camera)
    idx = next((i for i, im in enumerate(timeline) if im.sd_token == keyframe.image.sd_token), None)
    if idx is None:
        raise PropagationError(
            "TIMELINE_MISMATCH", f"Không thấy ảnh của {keyframe.frame_id} trong chuỗi camera của scene", 422
        )
    return timeline[idx + 1 :]


def propagate_from(
    store: WorkspaceStore,
    source: SequenceSource,
    config: AutoLabelConfig,
    keyframe_id: str,
    max_frames: int | None = None,
) -> PropagateResponse:
    keyframe = store.load_frame(keyframe_id)
    if keyframe is None:
        raise PropagationError("FRAME_NOT_FOUND", f"Không có frame {keyframe_id}", 404)
    if keyframe.status != "approved":
        raise PropagationError(
            "NOT_APPROVED", "Chỉ lan truyền từ frame đã approve (mọi quyết định của người đã chốt)", 409
        )
    video = store.load_video(keyframe.scene) if keyframe.scene.startswith("vid-") else None
    if video is not None and video.source == "images":
        resp = PropagateResponse(keyframe_id=keyframe_id, tracks_started=0)
        resp.stop_reason = "Bộ ảnh rời: các ảnh không liên tiếp nhau nên không lan truyền"
        return resp
    cfg = config.propagation
    max_frames = max_frames or cfg.max_keyframes
    images = _timeline_after(source, keyframe)

    tracks = init_tracks(keyframe, store.load_aux("lidar", keyframe_id))
    store.save_frame(keyframe)  # lưu track_id vừa gán cho object của keyframe
    resp = PropagateResponse(keyframe_id=keyframe_id, tracks_started=len(tracks))
    if not any(t.kind == "keep" for t in tracks):
        resp.stop_reason = "Keyframe không có object nào đã duyệt để lan truyền"
        return resp

    scene_frames = [f for f in store.list_frames() if f.scene == keyframe.scene]
    frame_of_sample = {f.sample_token: f.frame_id for f in scene_frames}
    overrides = sweep_overrides(scene_frames)
    tracker = Tracker(
        tracks, cfg, keyframe.image.width, keyframe.image.height, _motion(source, cfg), keyframe.image.path
    )
    at = now_iso()
    hops = 0
    for image in images:
        dets = overrides[image.sd_token] if image.sd_token in overrides else source.detections(image.sd_token)
        tracker.step(image, dets)
        if not image.is_keyframe:
            continue
        hops += 1
        if not any(t.kind == "keep" for t in tracker.alive):
            resp.stop_reason = "Mọi object đã dừng lan truyền (mất dấu, ra khỏi khung hoặc độ tin cậy thấp)"
            break
        frame_id = frame_of_sample.get(image.sample_token) or f"{keyframe.scene}_{keyframe.index + hops:03d}"
        frame = store.load_frame(frame_id) if image.sample_token in frame_of_sample else None
        if frame is None:
            resp.frames_skipped.append(PropagateSkip(frame_id=frame_id, reason="Frame chưa chạy auto-label"))
        elif frame.status != "auto":
            resp.stopped_at = frame_id
            resp.stop_reason = f"Dừng trước {frame_id}: frame đã có người mở hoặc duyệt"
            break
        else:
            result = apply_propagation(frame, keyframe_id, tracker.alive, config, store.load_aux("lidar", frame_id), at)
            store.save_frame(frame)
            resp.frames_updated.append(frame_id)
            resp.objects_propagated += result.propagated
            resp.objects_suppressed += result.suppressed
        if not any(t.kind == "keep" for t in tracker.alive):
            resp.stop_reason = "Mọi object đã dừng lan truyền (mất dấu, ra khỏi khung hoặc độ tin cậy thấp)"
            break
        if hops >= max_frames:
            resp.stop_reason = f"Đã đi đủ {max_frames} keyframe"
            break
    else:
        resp.stop_reason = "Hết scene"

    resp.tracks_alive = sum(t.kind == "keep" for t in tracker.alive)
    resp.images_without_detections = tracker.images_without_detections
    return resp


# ---------------------------------------------------------------------------
# Thí nghiệm "keyframe hoàn hảo"
# ---------------------------------------------------------------------------


def _gt_keyframe(frame: FrameRecord, gts: list[dict]) -> FrameRecord:
    """Keyframe giả lập: mọi GT (trừ object bị che gần hết) là nhãn người đã duyệt, track_id = instance_token."""
    objs = []
    for g in gts:
        if g["ignore"]:
            continue
        objs.append(
            LabelObject(
                object_id=g["instance_token"][:8],
                bbox=g["bbox"],
                label=g["label"],
                score=1.0,
                source="human",
                track_id=g["instance_token"],
                review=ReviewState(
                    status="approved",
                    action="ADD_BOX",
                    final_bbox=g["bbox"],
                    final_label=g["label"],
                    reviewer=PROPAGATION_REVIEWER,
                ),
            )
        )
    return frame.model_copy(update={"objects": objs, "status": "approved"})


def evaluate_propagation(
    store: WorkspaceStore,
    source: SequenceSource,
    config: AutoLabelConfig,
    scenes: list[str] | None = None,
    max_frames: int | None = None,
    thr: float = 0.5,
    start_every: int | None = None,
) -> dict:
    """Lan truyền GT của keyframe đầu mỗi scene rồi đo với GT cùng instance ở các keyframe sau.

    start_every=k: bắt đầu thêm từ keyframe thứ k, 2k, ... của mỗi scene (mỗi lần là một thí nghiệm độc lập), để có
    đủ object khi chỉ có vài scene.

    Với mỗi track ở mỗi keyframe đích:
      đúng      : IoU với GT của chính instance đó >= thr
      đổi ID    : sai, nhưng trùng GT của instance khác (track nhảy sang object khác)
      sai       : còn lại (trôi, hoặc instance đã biến mất mà track vẫn chạy)
    Instance vẫn hiện diện trong frame nhưng track đã dừng tính là "mất".
    """
    cfg = config.propagation
    max_frames = max_frames or cfg.max_keyframes
    frames = store.list_frames()
    by_scene: dict[str, list[FrameRecord]] = defaultdict(list)
    for f in frames:
        if scenes is None or f.scene in scenes:
            by_scene[f.scene].append(f)

    per_hop: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    calib = {"correct": [], "wrong": []}
    n_scenes = n_tracks = 0

    runs = []
    for _scene, scene_frames in sorted(by_scene.items()):
        scene_frames.sort(key=lambda f: f.index)
        step = start_every or len(scene_frames)
        runs += [(scene_frames, i) for i in range(0, max(1, len(scene_frames) - 1), step)]

    for scene_frames, start in runs:
        first = scene_frames[start]
        gts0 = store.load_aux("gt", first.frame_id) or []
        keyframe = _gt_keyframe(first, gts0)
        tracks = init_tracks(keyframe, store.load_aux("lidar", first.frame_id))
        if not tracks:
            continue
        n_scenes += 1
        n_tracks += len(tracks)
        frame_of_sample = {f.sample_token: f for f in scene_frames}
        tracker = Tracker(tracks, cfg, first.image.width, first.image.height, _motion(source, cfg), first.image.path)
        hops = 0
        for image in _timeline_after(source, first):
            tracker.step(image, source.detections(image.sd_token))
            if not image.is_keyframe:
                continue
            hops += 1
            frame = frame_of_sample.get(image.sample_token)
            if frame is not None:
                gts = [g for g in (store.load_aux("gt", frame.frame_id) or []) if not g["ignore"]]
                uv, _ = lidar_arrays(store.load_aux("lidar", frame.frame_id))
                _score_hop(per_hop[hops], calib, tracker.tracks, gts, uv, cfg, thr)
            if not tracker.alive or hops >= max_frames:
                break

    rows = []
    for hop in sorted(per_hop):
        s = per_hop[hop]
        n = int(s["outputs"])
        rows.append(
            {
                "hop": hop,
                "tracks_output": n,
                "correct": int(s["correct"]),
                "id_switch": int(s["id_switch"]),
                "wrong": int(s["wrong"]),
                "lost": int(s["lost"]),
                "correct_rate": round(s["correct"] / n, 4) if n else None,
                "mean_iou": round(s["iou_sum"] / n, 4) if n else None,
                "coverage": round(s["covered"] / s["present"], 4) if s["present"] else None,
                **{
                    k: int(s[k])
                    for k in ("outputs_emit", "correct_emit", "id_switch_emit", "wrong_emit", "missing_emit")
                },
            }
        )
    return {
        "match_iou": thr,
        "scenes": len(by_scene),
        "starts": n_scenes,
        "tracks_started": n_tracks,
        "per_hop": rows,
        "calibration": _calibration(calib, cfg.flag_below),
        "config": cfg.model_dump(),
    }


def _score_hop(
    acc: dict, calib: dict, tracks: list[Track], gts: list[dict], uv: np.ndarray | None, cfg, thr: float
) -> None:
    gt_by_inst = {g["instance_token"]: g for g in gts}
    alive_ids, emitted_ids = set(), set()
    for t in tracks:
        if not t.alive or t.kind != "keep":
            continue
        alive_ids.add(t.track_id)
        box = t.output_box()
        c = track_confidence(t, uv, cfg)
        own = gt_by_inst.get(t.track_id)
        v = iou(box, own["bbox"]) if own else 0.0
        # Sản phẩm chỉ ghi box khi track khớp detection ở keyframe đích (emit_coasting: false); đếm riêng phần đó ("_emit")
        emit = t.last_match is not None or cfg.emit_coasting
        if emit:
            acc["outputs_emit"] += 1
            emitted_ids.add(t.track_id)
        acc["outputs"] += 1
        acc["iou_sum"] += v
        if v >= thr:
            acc["correct"] += 1
            acc["correct_emit"] += emit
            calib["correct"].append(c)
            continue
        calib["wrong"].append(c)
        if any(iou(box, g["bbox"]) >= thr for inst, g in gt_by_inst.items() if inst != t.track_id):
            acc["id_switch"] += 1
            acc["id_switch_emit"] += emit
        else:
            acc["wrong"] += 1
            acc["wrong_emit"] += emit
    started = {t.track_id for t in tracks if t.kind == "keep"}
    for inst in gt_by_inst:
        if inst in started:
            acc["present"] += 1
            if inst in alive_ids:
                acc["covered"] += 1
            else:
                acc["lost"] += 1
            if inst not in emitted_ids:
                acc["missing_emit"] += 1  # vật còn đó mà sản phẩm không ghi nhãn lan truyền


def _calibration(calib: dict, flag_below: float) -> dict:
    """c_prop có tách được nhãn lan truyền đúng và sai không; cờ PROP_LOW_CONF bắt được bao nhiêu lỗi."""
    good, bad = np.array(calib["correct"]), np.array(calib["wrong"])
    flagged_bad = int((bad < flag_below).sum())
    flagged = flagged_bad + int((good < flag_below).sum())

    def mean(a):
        return round(float(a.mean()), 4) if len(a) else None

    return {
        "n_correct": int(len(good)),
        "n_wrong": int(len(bad)),
        "mean_c_prop_correct": mean(good),
        "mean_c_prop_wrong": mean(bad),
        "low_conf_flag_recall": round(flagged_bad / len(bad), 4) if len(bad) else None,
        "low_conf_flag_precision": round(flagged_bad / flagged, 4) if flagged else None,
        "flag_below": flag_below,
    }


def render_propagation_report(result: dict, meta: dict) -> str:
    lines = [
        "# Đánh giá lan truyền nhãn 2D — thí nghiệm keyframe hoàn hảo",
        "",
        f"- Dữ liệu: {meta.get('camera')} · nuScenes {meta.get('version')} · {result['scenes']} scene, "
        f"{result['tracks_started']} object ở keyframe đầu",
        "- Cách làm: GT ở keyframe đầu mỗi scene đóng vai nhãn người đã duyệt; lan truyền qua mọi ảnh 12Hz; "
        f"so box ra ở mỗi keyframe sau với GT cùng `instance_token` (đúng nếu IoU ≥ {result['match_iou']}).",
        "- GT là hộp bao box 3D chiếu xuống nên rộng hơn box detector: IoU tuyệt đối thấp hơn thực tế, "
        "chỉ nên so tương đối giữa các cấu hình.",
        "",
        "| Keyframe thứ | Track ra | Đúng | Đổi ID | Sai | Mất | Tỉ lệ đúng | IoU TB | Độ phủ |",
        "| --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for r in result["per_hop"]:
        lines.append(
            f"| {r['hop']} | {r['tracks_output']} | {r['correct']} | {r['id_switch']} | {r['wrong']} | {r['lost']} "
            f"| {_fmt(r['correct_rate'])} | {_fmt(r['mean_iou'])} | {_fmt(r['coverage'])} |"
        )
    c = result["calibration"]
    lines += [
        "",
        "Độ phủ = trong số object của keyframe đầu còn xuất hiện ở frame đó, bao nhiêu % vẫn còn track.",
        "",
        "## c_prop có đáng tin không",
        "",
        "| Chỉ số | Giá trị |",
        "| :-- | --: |",
        f"| c_prop TB của nhãn đúng | {_fmt(c['mean_c_prop_correct'])} |",
        f"| c_prop TB của nhãn sai | {_fmt(c['mean_c_prop_wrong'])} |",
        f"| PROP_LOW_CONF bắt được lỗi (recall, c_prop < {c['flag_below']}) | {_fmt(c['low_conf_flag_recall'])} |",
        f"| PROP_LOW_CONF đúng chỗ (precision) | {_fmt(c['low_conf_flag_precision'])} |",
        "",
        "Nếu c_prop TB của nhãn đúng không cao hơn rõ nhãn sai, chỉnh `propagation.weights` trong "
        "`configs/autolabel.yaml` rồi chạy lại.",
    ]
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"
