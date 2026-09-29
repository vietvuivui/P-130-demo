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
from src.services.nuscenes_data import CameraFrame, NuScenesMini
from src.services.review import now_iso
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore

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
    ):
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

    def process(self, scene: str, index: int, sample_token: str, overwrite: bool = False) -> FrameRecord | None:
        frame = self.data.camera_frame(sample_token, self.config.camera, self.config.sweep_offsets, scene, index)
        existing = self.store.load_frame(frame.frame_id)
        # Không ghi đè frame người đã bắt đầu duyệt
        if existing and (existing.status != "auto" or not overwrite):
            return None

        t0 = time.perf_counter()
        uv, depth = (
            self.data.lidar_in_image(frame, self.config.qa.lidar.min_depth_m) if frame.lidar_sd_token else (None, None)
        )
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
        )
        record.autolabel_s, record.autolabel_run = round(time.perf_counter() - t0, 3), self.run_id
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
) -> FrameRecord:
    """Bước 2 -> 3 cho một keyframe bất kỳ (scene nuScenes hoặc video tải lên): detect keyframe + sweep, QA Agent."""
    offsets = sorted(sweeps)
    images = [(image.sd_token, image_file(image.path))] + [
        (sweeps[o].sd_token, image_file(sweeps[o].path)) for o in offsets
    ]
    det_cfg = config.detection
    # Ngưỡng giữ box áp cho cả keyframe lẫn sweep để check temporal/tracking nhất quán với box người thấy
    detections = [[d for d in dets if det_cfg.keep(d.label, d.score)] for dets in ensemble.detect_batch(images)]
    key_dets = detections[0]
    sweep_dets = dict(zip(offsets, detections[1:], strict=True))

    objects = [
        LabelObject(
            object_id=str(i + 1),
            bbox=d.bbox,
            label=d.label,
            score=d.score,
            models=d.models,
            alternatives=d.alternatives,
            mask=d.mask,
        )
        for i, d in enumerate(key_dets)
    ]
    state = qa_agent.invoke(
        {
            "config": config,
            "image_size": (image.width, image.height),
            "intrinsic": intrinsic,
            "key_timestamp": image.timestamp,
            "objects": objects,
            "sweeps": {o: {"timestamp": sweeps[o].timestamp, "detections": sweep_dets[o]} for o in offsets},
            "lidar_uv": uv,
            "lidar_depth": depth,
        }
    )
    return FrameRecord(
        frame_id=frame_id,
        sample_token=sample_token,
        scene=scene,
        index=index,
        camera=camera,
        image=image,
        intrinsic=intrinsic,
        sweeps=[SweepInfo(offset=o, detections=sweep_dets[o], **sweeps[o].model_dump()) for o in offsets],
        detectors=ensemble.names,
        has_lidar=uv is not None,
        frame_risk=state["frame_risk"],
        objects=state["reviewed"],
        created_at=now_iso(),
    )


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
