"""Áp dụng lại cài đặt (ngưỡng, QA temporal, tính lại score) cho các frame chưa ai đụng tới — nút trên tab ⚙ Cài đặt.

Dùng cache detection nên nhanh (không phải chạy lại model, trừ khi cache thiếu). Không bao giờ ghi đè frame người đã mở /
duyệt / trả lại, frame đã nhận nhãn lan truyền, hay frame có sweep đã sửa.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import FrameRecord, ImageInfo
from src.services.store import WorkspaceStore


def skip_reason(f: FrameRecord) -> str | None:
    if f.status != "auto":
        return "opened"
    if f.propagated_from:
        return "propagated"
    if any(s.boxes is not None for s in f.sweeps):
        return "sweep_edited"
    return None


def relabel_auto_frames(
    store: WorkspaceStore,
    config: AutoLabelConfig,
    ensemble,
    dataroot: str | Path,
    version: str,
    progress: Callable[[float, str], None] = lambda p, m: None,
) -> dict:
    from src.services.pipeline import AutoLabelPipeline, label_keyframe
    from src.services.productivity import new_run_id

    frames = sorted(store.list_frames(), key=lambda f: (f.scene, f.index))
    skipped = {"opened": 0, "propagated": 0, "sweep_edited": 0}
    todo = []
    for f in frames:
        reason = skip_reason(f)
        if reason:
            skipped[reason] += 1
        else:
            todo.append(f)
    pipeline = None
    run_id = new_run_id("relabel2d")
    done = 0
    for n, f in enumerate(todo, start=1):
        progress((n - 1) / max(len(todo), 1), f"{n - 1}/{len(todo)} frame")
        video = store.load_video(f.scene) if f.scene.startswith("vid-") else None
        t0 = time.perf_counter()
        if video is None:  # scene nuScenes: chạy lại đúng như lúc auto-label (LiDAR đầy đủ cho QA)
            if pipeline is None:
                from src.services.nuscenes_data import NuScenesMini

                pipeline = AutoLabelPipeline(NuScenesMini(dataroot, version), store, config)
                pipeline.ensemble = ensemble
                pipeline.run_id = run_id
            record = pipeline.process(f.scene, f.index, f.sample_token, overwrite=True)
        else:  # video / bộ ảnh tải lên: không có LiDAR, sweep giữ như lúc cắt frame
            record = label_keyframe(
                ensemble,
                config,
                frame_id=f.frame_id,
                sample_token=f.sample_token,
                scene=f.scene,
                index=f.index,
                camera=f.camera,
                image=f.image,
                sweeps={s.offset: ImageInfo(**s.model_dump(include=set(ImageInfo.model_fields))) for s in f.sweeps},
                image_file=store.resolve,
                intrinsic=f.intrinsic,
                prev=store.load_frame(f"{f.scene}_{f.index - 1:03d}", copy=False) if f.index > 0 else None,
            )
            record.autolabel_s, record.autolabel_run = round(time.perf_counter() - t0, 3), run_id
            store.save_frame(record)
        done += record is not None
    progress(1.0, f"{done}/{len(todo)} frame")
    return {"relabeled": done, "total": len(frames), "skipped": skipped}
