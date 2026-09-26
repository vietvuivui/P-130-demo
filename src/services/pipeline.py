"""Pipeline 1 -> 3: nạp frame nuScenes, detect 2D (keyframe + sweep), chạy QA Agent, lưu kết quả."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.agents.graph import qa_agent
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, ImageInfo, LabelObject, SweepInfo
from src.services.detectors import DetectorEnsemble
from src.services.nuscenes_data import CameraFrame, NuScenesMini
from src.services.review import now_iso
from src.services.store import WorkspaceStore

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

    def run(self, scenes: list[str] | None = None, limit: int | None = None, overwrite: bool = False) -> list[str]:
        keyframes = self.data.keyframes(scenes)[:limit]
        done = []
        for n, (scene, index, token) in enumerate(keyframes, start=1):
            t0 = time.perf_counter()
            record = self.process(scene, index, token, overwrite=overwrite)
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

        offsets = sorted(frame.sweeps)
        images = [(frame.image.sd_token, self.data.dataroot / frame.image.path)] + [
            (frame.sweeps[o].sd_token, self.data.dataroot / frame.sweeps[o].path) for o in offsets
        ]
        detections = self.ensemble.detect_batch(images)
        key_dets = detections[0]
        sweep_dets = dict(zip(offsets, detections[1:], strict=True))

        uv, depth = (
            self.data.lidar_in_image(frame, self.config.qa.lidar.min_depth_m) if frame.lidar_sd_token else (None, None)
        )

        objects = [
            LabelObject(
                object_id=str(i + 1),
                bbox=d.bbox,
                label=d.label,
                score=d.score,
                models=d.models,
                alternatives=d.alternatives,
            )
            for i, d in enumerate(key_dets)
        ]
        state = qa_agent.invoke(
            {
                "config": self.config,
                "image_size": (frame.image.width, frame.image.height),
                "intrinsic": frame.intrinsic.tolist(),
                "key_timestamp": frame.image.timestamp,
                "objects": objects,
                "sweeps": {o: {"timestamp": frame.sweeps[o].timestamp, "detections": sweep_dets[o]} for o in offsets},
                "lidar_uv": uv,
                "lidar_depth": depth,
            }
        )

        record = FrameRecord(
            frame_id=frame.frame_id,
            sample_token=sample_token,
            scene=scene,
            index=index,
            camera=frame.camera,
            image=ImageInfo(**vars(frame.image)),
            intrinsic=frame.intrinsic.tolist(),
            sweeps=[SweepInfo(offset=o, detections=sweep_dets[o], **vars(frame.sweeps[o])) for o in offsets],
            detectors=self.ensemble.names,
            has_lidar=uv is not None,
            frame_risk=state["frame_risk"],
            objects=state["reviewed"],
            created_at=now_iso(),
        )
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


def image_path(dataroot: str | Path, record: FrameRecord, offset: int = 0) -> Path:
    if offset == 0:
        return Path(dataroot) / record.image.path
    for s in record.sweeps:
        if s.offset == offset:
            return Path(dataroot) / s.path
    raise KeyError(offset)
