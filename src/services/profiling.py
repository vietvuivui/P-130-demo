"""Đo một frame tốn thời gian ở đâu, từ đầu (không dùng cache): `python -m src.cli profile --limit 10`.

Chạy auto-label 2D cho N keyframe vào một workspace tạm (cache detection trống, nên detect chạy thật), rồi làm mờ
ảnh keyframe + sweep như khi mở frame trên web lần đầu. In bảng: giây mỗi frame cho từng bước, % tổng, nạp model.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from src.models.qa_config import AutoLabelConfig


def device_name() -> str:
    try:
        import torch

        return f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU (không có GPU)"
    except Exception:
        return "không rõ (chưa cài torch)"


def profile(data, config: AutoLabelConfig, scenes=None, limit: int = 10, privacy: bool = True) -> dict:
    from src.services import timing
    from src.services.pipeline import AutoLabelPipeline, image_path
    from src.services.store import WorkspaceStore

    tmp = Path(tempfile.mkdtemp(prefix="autolabel_profile_"))
    try:
        store = WorkspaceStore(tmp)
        pipe = AutoLabelPipeline(data, store, config)
        pipe.run_id = "profile"
        keys = []  # chỉ keyframe có ảnh trên máy (bản trainval tải một phần blob)
        for scene, index, token in data.keyframes(scenes):
            sd = data._keyframe_data.get(token, {}).get(config.camera)
            if sd and (Path(data.dataroot) / data.sample_data[sd]["filename"]).exists():
                keys.append((scene, index, token))
                if len(keys) >= limit:
                    break
        load0 = timing.snapshot().get("load_model", {}).get("total_s", 0.0)
        wall, frames = [], []
        for scene, index, token in keys:
            t0 = time.perf_counter()
            rec = pipe.process(scene, index, token, overwrite=True)
            wall.append(time.perf_counter() - t0)
            if rec is not None:
                frames.append(rec)
        load_s = timing.snapshot().get("load_model", {}).get("total_s", 0.0) - load0
        blur = []
        if privacy and config.privacy.enabled:
            from src.services.privacy import anonymized_path

            for f in frames:
                for off in [0] + [s.offset for s in f.sweeps]:
                    src = image_path(data.dataroot, f, off, workspace=tmp)
                    t0 = time.perf_counter()
                    anonymized_path(tmp, src, config.privacy)
                    blur.append(time.perf_counter() - t0)
        timings = [f.autolabel_timing or {} for f in frames]
        # Frame đầu gồm cả nạp model: tách ra để số TB không bị kéo lên
        if timings and load_s:
            timings[0] = {**timings[0], "detect": max(0.0, timings[0].get("detect", 0.0) - load_s)}
        return {
            "device": device_name(),
            "frames": len(frames),
            "images_per_frame": round(sum(t.get("n_model_images", 0) for t in timings) / max(len(timings), 1), 1),
            "load_model_s": round(load_s, 2),
            "rows": timing.summarize(timings),
            "mean_frame_s": round((sum(wall) - load_s) / max(len(wall), 1), 3),
            "privacy_per_image_s": round(sum(blur) / len(blur), 3) if blur else None,
            "privacy_images_per_frame": round(len(blur) / max(len(frames), 1), 1) if blur else 0,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def print_profile(r: dict) -> None:
    print(f"\nThiết bị: {r['device']} · {r['frames']} keyframe · {r['images_per_frame']} ảnh detect / keyframe")
    print(f"Nạp model (một lần mỗi lần mở server): {r['load_model_s']:.1f} s")
    print(f"\n{'Bước':<40} {'giây/frame':>10} {'% tổng':>8}")
    for row in r["rows"]:
        print(f"{row['name']:<40} {row['mean_s']:>10.3f} {row['share'] * 100:>7.1f}%")
    print(f"{'Tổng auto-label 2D':<40} {r['mean_frame_s']:>10.3f}")
    if r["privacy_per_image_s"] is not None:
        per_frame = r["privacy_per_image_s"] * r["privacy_images_per_frame"]
        print(f"\nLàm mờ mặt / biển số khi mở frame trên web lần đầu: {r['privacy_per_image_s']:.3f} s/ảnh × "
              f"{r['privacy_images_per_frame']:.0f} ảnh = {per_frame:.2f} s/frame (lần sau lấy từ cache)")  # fmt: skip
    print(
        "\nPhần 3D (dự đoán 4 mô hình LiDAR, kiểm chứng bằng camera): xem tab Metrics → 'Thời gian từng bước' của dự án."
    )
