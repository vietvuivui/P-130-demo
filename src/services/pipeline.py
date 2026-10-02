"""Pipeline 1 -> 3: nạp frame nuScenes, detect 2D (keyframe + sweep), chạy QA Agent, lưu kết quả."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.agents.graph import qa_agent
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, ImageInfo, LabelObject, SweepInfo
from src.services.detectors import DetectorEnsemble
from src.services.flow import FlowProvider
from src.services.lidar2d import SOURCE as LIDAR3D
from src.services.lidar2d import merge_detections, project_boxes
from src.services.nuscenes_data import CameraFrame, NuScenesMini
from src.services.review import now_iso
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore
from src.services.temporal_fusion import rescore
from src.services.timing import Stopwatch

log = logging.getLogger(__name__)

# Số điểm LiDAR tối đa lưu cho UI vẽ overlay (đủ nhìn, file nhỏ)
MAX_UI_POINTS = 8000


class AutoLabelPipeline:
    def __init__(
        self,
        data: NuScenesMini,
        store: WorkspaceStore,
        config: AutoLabelConfig,
        detectors: list[str] | None = None,
        preds3d: dict[str, list[dict]] | None = None,
    ):
        self.preds3d = preds3d  # sample_token -> box 3D nuScenes (hệ toàn cục); None: chỉ dùng detector ảnh
        self.data = data
        self.store = store
        self.config = config
        self.ensemble = DetectorEnsemble(config, store.root / "cache" / "detections", detectors)
        self.run_id: str | None = None  # phiên auto-label (FR-27), đặt khi run() bắt đầu

    def run(self, scenes: list[str] | None = None, limit: int | None = None, overwrite: bool = False,
            progress=None) -> list[str]:  # fmt: skip
        from src.services.productivity import new_run_id

        keyframes = self.data.keyframes(scenes)[:limit]
        self.run_id = new_run_id("run2d")
        done = []
        for n, (scene, index, token) in enumerate(keyframes, start=1):
            t0 = time.perf_counter()
            record = self.process(scene, index, token, overwrite=overwrite)
            if progress:
                progress(n, len(keyframes))
            if record is None:
                continue
            done.append(record.frame_id)
            log.info(
                "[%d/%d] %s: %d object, frame_risk=%.2f (%.1fs)",
                n,
                len(keyframes),
                record.frame_id,
                len(record.objects),
                record.frame_risk,
                time.perf_counter() - t0,
            )
        return done

    def process(self, scene: str, index: int, sample_token: str, overwrite: bool = False,
                keep_propagated: bool = False) -> FrameRecord | None:  # fmt: skip
        frame = self.data.camera_frame(sample_token, self.config.camera, self.config.sweep_offsets, scene, index)
        existing = self.store.load_frame(frame.frame_id)
        # Không ghi đè frame người đã bắt đầu duyệt
        if existing and (existing.status != "auto" or not overwrite):
            return None
        if existing and keep_propagated and existing.prelabel is not None:  # đã nhận nhãn lan truyền từ frame đã duyệt
            return None
        boxes3d, l3d = None, self.config.detection.lidar3d
        if self.preds3d is not None and l3d.enabled:
            boxes3d = project_boxes(
                self.preds3d.get(sample_token, []), self.data.cam_from_global(frame.image.sd_token), frame.intrinsic,
                frame.image.width, frame.image.height, l3d.min_score,
                ego_from_global=np.linalg.inv(self.data._global_from_ego(frame.image.sd_token)),
            )  # fmt: skip

        t0 = time.perf_counter()
        uv, depth = (
            self.data.lidar_in_image(frame, self.config.qa.lidar.min_depth_m) if frame.lidar_sd_token else (None, None)
        )
        t_lidar = time.perf_counter() - t0
        record = label_keyframe(
            self.ensemble,
            self.config,
            frame_id=frame.frame_id,
            sample_token=sample_token,
            scene=scene,
            index=index,
            camera=frame.camera,
            image=ImageInfo(**vars(frame.image)),
            sweeps={o: ImageInfo(**vars(ref)) for o, ref in frame.sweeps.items()},
            image_file=lambda path: self.data.dataroot / path,
            intrinsic=frame.intrinsic.tolist(),
            uv=uv,
            depth=depth,
            boxes3d=boxes3d,
        )
        record.autolabel_s, record.autolabel_run = round(time.perf_counter() - t0, 3), self.run_id
        if frame.lidar_sd_token:
            record.autolabel_timing = {"lidar": round(t_lidar, 3), **(record.autolabel_timing or {})}
        self.store.save_frame(record)
        self._save_aux(frame, uv, depth)
        return record

    def _save_aux(self, frame: CameraFrame, uv: np.ndarray | None, depth: np.ndarray | None) -> None:
        if uv is not None and len(uv):
            idx = np.random.default_rng(0).permutation(len(uv))[:MAX_UI_POINTS]
            self.store.save_aux(
                "lidar",
                frame.frame_id,
                {
                    "u": np.round(uv[idx, 0]).astype(int).tolist(),
                    "v": np.round(uv[idx, 1]).astype(int).tolist(),
                    "d": np.round(depth[idx], 1).tolist(),
                },
            )
        self.store.save_aux("gt", frame.frame_id, self.data.gt_boxes_2d(frame, self.config.gt_category_map))


def label_keyframe(
    ensemble: DetectorEnsemble,
    config: AutoLabelConfig,
    *,
    frame_id: str,
    sample_token: str,
    scene: str,
    index: int,
    camera: str,
    image: ImageInfo,
    sweeps: dict[int, ImageInfo],
    image_file: Callable[[str], Path],
    intrinsic: list[list[float]],
    uv: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    boxes3d: list[dict] | None = None,
) -> FrameRecord:
    """Bước 2 -> 3 cho một keyframe bất kỳ (scene nuScenes hoặc video tải lên): detect keyframe + sweep, QA Agent."""
    offsets = sorted(sweeps)
    images = [(image.sd_token, image_file(image.path))] + [
        (sweeps[o].sd_token, image_file(sweeps[o].path)) for o in offsets
    ]
    det_cfg, tcfg = config.detection, config.qa.temporal
    sw = Stopwatch()
    n0 = getattr(ensemble, "model_images", 0)
    with sw("detect"):
        raw = ensemble.detect_batch(images)
    sw.add("n_model_images", getattr(ensemble, "model_images", 0) - n0)
    # Ngưỡng giữ box áp cho sweep để check temporal/tracking nhất quán với box người thấy
    sweep_min = tcfg.sweep_min_score

    def keep_sweep(d) -> bool:
        return d.score >= sweep_min if sweep_min is not None else det_cfg.keep(d.label, d.score)

    sweep_dets = {o: [d for d in dets if keep_sweep(d)] for o, dets in zip(offsets, raw[1:], strict=True)}
    # Box score thấp (chưa qua ngưỡng giữ) chỉ làm bằng chứng cho RECOVERED_BY_TRACK (qa.temporal.recover_weak_min_score)
    weak_lo = tcfg.recover_weak_min_score
    sweep_weak = {
        o: [strip_mask(d) for d in dets if not keep_sweep(d) and weak_lo is not None and d.score >= weak_lo]
        for o, dets in zip(offsets, raw[1:], strict=True)
    }
    with sw("flow"):
        warped = sweep_warps(image, sweeps, sweep_dets, image_file, tcfg) if offsets else {}
    # Score keyframe tính lại theo sweep trước khi lọc ngưỡng: vật thấy ổn định được giữ, box chỉ loé lên bị bỏ
    key_all = rescore(raw[0], sweep_dets, warped, tcfg.rescore, tcfg.match_iou)
    if boxes3d is not None:  # box 3D chiếu xuống ảnh (lidar2d.py): [{bbox, label, score}]
        key_all = merge_detections(key_all, boxes3d, det_cfg.lidar3d.match_iou, det_cfg.lidar3d.camera_only_scale)
    key_dets = sorted((d for d in key_all if det_cfg.keep(d.label, d.score)), key=lambda d: -d.score)
    weak_key = [d for d in key_all if not det_cfg.keep(d.label, d.score) and weak_lo is not None and d.score >= weak_lo]

    objects = [
        LabelObject(
            object_id=str(i + 1),
            bbox=d.bbox,
            label=d.label,
            score=d.score,
            det_score=d.det_score,
            models=d.models,
            alternatives=d.alternatives,
            mask=d.mask,
            box3d=d.box3d,
        )
        for i, d in enumerate(key_dets)
    ]
    qa_t0 = time.perf_counter()
    state = qa_agent.invoke(
        {
            "config": config,
            "image_size": (image.width, image.height),
            "intrinsic": intrinsic,
            "key_timestamp": image.timestamp,
            "objects": objects,
            "sweeps": {o: sweep_state(sweeps[o].timestamp, sweep_dets[o], warped.get(o), sweep_weak[o]) for o in offsets},
            "weak_key": weak_key,
            "lidar_uv": uv,
            "lidar_depth": depth,
        }
    )
    sw.add("qa", time.perf_counter() - qa_t0)
    if not tcfg.flow:
        sw.t.pop("flow", None)
    return FrameRecord(
        autolabel_timing=sw.result(),
        frame_id=frame_id,
        sample_token=sample_token,
        scene=scene,
        index=index,
        camera=camera,
        image=image,
        intrinsic=intrinsic,
        sweeps=[SweepInfo(offset=o, detections=sweep_dets[o], weak=sweep_weak[o], **sweeps[o].model_dump()) for o in offsets],
        weak=[strip_mask(d) for d in weak_key],
        detectors=ensemble.names + ([LIDAR3D] if boxes3d is not None else []),
        has_lidar=uv is not None,
        frame_risk=state["frame_risk"],
        objects=state["reviewed"],
        created_at=now_iso(),
    )


def sweep_state(timestamp: int, dets: list, warped: list | None, weak: list | None = None) -> dict:
    state = {"timestamp": timestamp, "detections": dets}
    if warped is not None:
        state["warped"] = warped
    if weak:
        state["weak"] = weak
    return state


def strip_mask(d):
    """Detection chỉ để làm bằng chứng: bỏ mask cho file frame gọn."""
    return d.model_copy(update={"mask": None}) if d.mask is not None else d


def sweep_warps(
    image: ImageInfo,
    sweeps: dict[int, ImageInfo],
    sweep_dets: dict[int, list],
    image_file: Callable[[str], Path],
    tcfg,
) -> dict[int, list[list[float]]]:
    """Box của mỗi sweep dời về thời điểm keyframe bằng optical flow sweep -> keyframe. Rỗng nếu tắt flow."""
    if not tcfg.flow:
        return {}
    flows = FlowProvider(lambda p: image_file(p), tcfg.flow_scale)
    out = {}
    for o, dets in sweep_dets.items():
        if not dets:
            out[o] = []
            continue
        field = flows.between(sweeps[o].path, image.path)
        if field is not None:
            out[o] = [[round(v, 1) for v in field.warp_box(d.bbox)] for d in dets]
    return out


def resolve_image(dataroot: str | Path, workspace: str | Path | None, path: str) -> Path:
    """Ảnh nuScenes tương đối so với dataroot; frame của video tải lên có tiền tố '@workspace/'."""
    if path.startswith(WORKSPACE_PREFIX) and workspace is not None:
        return Path(workspace) / path.removeprefix(WORKSPACE_PREFIX)
    return Path(dataroot) / path


def image_path(dataroot: str | Path, record: FrameRecord, offset: int = 0, workspace: str | Path | None = None) -> Path:
    if offset == 0:
        return resolve_image(dataroot, workspace, record.image.path)
    for s in record.sweeps:
        if s.offset == offset:
            return resolve_image(dataroot, workspace, s.path)
    raise KeyError(offset)
