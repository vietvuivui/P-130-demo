"""CLI chạy batch.

python -m src.cli run --limit 20                       # auto-label + QA Agent cho 20 keyframe đầu
python -m src.cli run --scenes scene-0061 scene-0103   # theo scene
python -m src.cli run --detectors yolo_world grounding_dino --overwrite
python -m src.cli evaluate                             # mAP + flag recall/precision -> eval/results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import get_settings
from src.models.qa_config import load_autolabel_config


def cmd_run(args) -> None:
    from src.services.nuscenes_data import NuScenesMini
    from src.services.pipeline import AutoLabelPipeline
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    pipeline = AutoLabelPipeline(
        NuScenesMini(settings.nuscenes_dataroot, settings.nuscenes_version),
        WorkspaceStore(settings.workspace_dir),
        config,
        detectors=args.detectors,
    )
    done = pipeline.run(scenes=args.scenes, limit=args.limit, overwrite=args.overwrite)
    print(f"Xong {len(done)} frame -> {settings.workspace_dir}")


def cmd_evaluate(args) -> None:
    from src.services.evaluation import evaluate_map, evaluate_qa, render_report
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    store = WorkspaceStore(settings.workspace_dir)
    frames = store.list_frames()
    if not frames:
        raise SystemExit("Workspace trống, chạy `python -m src.cli run` trước")
    gt = {f.frame_id: store.load_aux("gt", f.frame_id) or [] for f in frames}

    result = {
        "n_frames": len(frames),
        "camera": config.camera,
        "version": settings.nuscenes_version,
        "detectors": sorted({d for f in frames for d in f.detectors}),
        "map50": evaluate_map(frames, gt, 0.5),
        "map70": evaluate_map(frames, gt, 0.7),
        "qa": evaluate_qa(frames, gt, 0.5),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autolabel2d_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(result)
    (out / "autolabel2d_eval.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Auto-label 2D + QA Agent cho các keyframe")
    run.add_argument("--scenes", nargs="*", help="Tên scene, ví dụ scene-0061 (mặc định: tất cả)")
    run.add_argument("--limit", type=int, help="Số keyframe tối đa")
    run.add_argument("--detectors", nargs="*", help="Ghi đè detection.detectors trong config")
    run.add_argument("--overwrite", action="store_true", help="Chạy lại frame chưa ai duyệt (dùng cache detection)")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("evaluate", help="mAP pre-label + flag recall/precision của QA Agent so với GT")
    ev.add_argument("--out", default="eval/results")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
