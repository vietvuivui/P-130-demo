"""Đo tốc độ detector trên cùng một bộ ảnh (từng ảnh một, sau 2 ảnh khởi động). Ghi JSON cho notebook so sánh.

    python scripts/bench_detectors.py --detectors yolo_world yoloe yolo26 --images 24 \\
        --out eval/results/compare/speed.json

Ảnh lấy đều trong các keyframe CAM_FRONT của NUSCENES_DATAROOT (đọc từ .env / biến môi trường).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from src.config import get_settings
    from src.models.qa_config import load_autolabel_config
    from src.services.detectors import build_detector
    from src.services.nuscenes_data import NuScenesMini

    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--config", default=None, help="Mặc định: AUTOLABEL_CONFIG")
    ap.add_argument("--out", type=Path, default=Path("eval/results/compare/speed.json"))
    args = ap.parse_args()

    s = get_settings()
    config = load_autolabel_config(args.config or s.autolabel_config)
    data = NuScenesMini(s.nuscenes_dataroot, s.nuscenes_version)
    keys = data.keyframes()
    step = max(1, len(keys) // args.images)
    paths = []
    for _scene, _idx, sample in keys[::step][: args.images]:
        sd = data._keyframe_data[sample][config.camera]
        paths.append(Path(s.nuscenes_dataroot) / data.sample_data[sd]["filename"])

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "device": torch.cuda.get_device_name(0)
        if device == "cuda"
        else f"CPU ({platform.processor() or 'x86'}, {torch.get_num_threads()} threads)",
        "n_images": len(paths),
        "detectors": {},
    }
    for name in args.detectors:
        det = build_detector(name, config)
        det.detect(paths[:2])  # khởi động
        times = []
        for p in paths:
            t = time.perf_counter()
            det.detect([p])
            times.append(time.perf_counter() - t)
        result["detectors"][name] = {
            "mean_s": round(statistics.mean(times), 3),
            "median_s": round(statistics.median(times), 3),
            "imgsz": getattr(getattr(config.detection, name, None), "imgsz", None),
            "weights": Path(str(getattr(getattr(config.detection, name, None), "weights", ""))).name,
        }
        print(name, result["detectors"][name])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
