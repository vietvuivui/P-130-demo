"""CLI chạy batch.

python -m src.cli run --limit 20                       # auto-label + QA Agent cho 20 keyframe đầu
python -m src.cli run --scenes scene-0061 scene-0103   # theo scene
python -m src.cli run --detectors yoloe yolo26 --overwrite
python -m src.cli evaluate                             # mAP + flag recall/precision -> eval/results/

Lan truyền nhãn trên video:
python -m src.cli detect-sweeps --scenes scene-0061    # detect mọi ảnh CAM_FRONT 12Hz của scene (GPU, có cache)
python -m src.cli propagate scene-0061_000             # lan truyền từ frame đã approve sang các keyframe sau
python -m src.cli eval-propagation                     # thí nghiệm keyframe hoàn hảo -> eval/results/
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
    from src.services.evaluation import evaluate_map, evaluate_pr, evaluate_qa, render_report
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    store = WorkspaceStore(settings.workspace_dir)
    # Chỉ frame có GT: video mp4 tải lên và dữ liệu nuScenes chưa gán nhãn (không có sample_annotation) không tính
    gt = {f.frame_id: g for f in store.list_frames() if (g := store.load_aux("gt", f.frame_id)) is not None}
    frames = [f for f in store.list_frames() if f.frame_id in gt]
    if not any(gt.values()):
        raise SystemExit(
            "Không có frame nào có GT để đánh giá (dữ liệu chưa gán nhãn, hoặc chưa `python -m src.cli run`). "
            "Với dữ liệu mới, đo bằng tab Metrics trong UI (thời gian duyệt, tỉ lệ nhãn phải sửa) sau khi duyệt, "
            "hoặc export các frame đã duyệt rồi dùng làm GT."
        )

    result = {
        "n_frames": len(frames),
        "camera": config.camera,
        "version": settings.nuscenes_version,
        "detectors": sorted({d for f in frames for d in f.detectors}),
        "map50": evaluate_map(frames, gt, 0.5),
        "map70": evaluate_map(frames, gt, 0.7),
        "pr": evaluate_pr(frames, gt, 0.5),
        "qa": evaluate_qa(frames, gt, 0.5),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autolabel2d_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(result)
    (out / "autolabel2d_eval.md").write_text(report, encoding="utf-8")
    print(report)


def _sequence_source(settings, config):
    from src.services.detectors import DetectorEnsemble
    from src.services.nuscenes_data import NuScenesMini
    from src.services.sequence import NuScenesSequenceSource

    data = NuScenesMini(settings.nuscenes_dataroot, settings.nuscenes_version)
    ensemble = DetectorEnsemble(config, Path(settings.workspace_dir) / "cache" / "detections")
    return data, ensemble, NuScenesSequenceSource(data, ensemble)


def cmd_detect_sweeps(args) -> None:
    """Chạy detector trên mọi ảnh camera của scene để tracker có detection ở từng ảnh 12Hz.

    Pipeline `run` đã cache sweep t-2..t+2 quanh mỗi keyframe, nên lệnh này chỉ phải chạy thêm
    phần còn thiếu giữa hai keyframe. Không chạy thì lan truyền vẫn được, chỉ kém chính xác hơn.
    """
    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    data, ensemble, source = _sequence_source(settings, config)
    if args.detectors:
        ensemble.names = args.detectors
    scenes = args.scenes or sorted(s["name"] for s in data.scene.values())
    for scene in scenes:
        images = source.timeline(scene, config.camera)
        todo = [im for im in images if ensemble.load_cached(im.sd_token) is None]
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            ensemble.detect_batch([(im.sd_token, data.dataroot / im.path) for im in chunk])
        print(f"{scene}: {len(images)} ảnh, detect thêm {len(todo)}")


def cmd_propagate(args) -> None:
    from src.services.sequence import propagate_from
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    if args.detectors:  # đọc cache detection của đúng detector đã chạy cho workspace này
        config.detection.detectors = args.detectors
    _, _, source = _sequence_source(settings, config)
    resp = propagate_from(WorkspaceStore(settings.workspace_dir), source, config, args.frame_id, args.max_frames)
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))


def cmd_eval_propagation(args) -> None:
    from src.services.sequence import evaluate_propagation, render_propagation_report
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    if args.detectors:  # đọc cache detection của đúng detector đã chạy cho workspace này
        config.detection.detectors = args.detectors
    _, _, source = _sequence_source(settings, config)
    store = WorkspaceStore(settings.workspace_dir)
    if not store.frame_ids():
        raise SystemExit("Workspace trống, chạy `python -m src.cli run` trước")
    result = evaluate_propagation(store, source, config, scenes=args.scenes, max_frames=args.max_frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "propagation_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_propagation_report(result, {"camera": config.camera, "version": settings.nuscenes_version})
    (out / "propagation_eval.md").write_text(report, encoding="utf-8")
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

    ds = sub.add_parser("detect-sweeps", help="Detect mọi ảnh camera 12Hz của scene (cho lan truyền)")
    ds.add_argument("--scenes", nargs="*", help="Tên scene (mặc định: tất cả)")
    ds.add_argument("--detectors", nargs="*", help="Ghi đè detection.detectors trong config")
    ds.add_argument("--batch", type=int, default=16, help="Số ảnh mỗi lô")
    ds.set_defaults(func=cmd_detect_sweeps)

    pr = sub.add_parser("propagate", help="Lan truyền nhãn từ một frame đã approve")
    pr.add_argument("frame_id", help="Ví dụ scene-0061_000")
    pr.add_argument("--max-frames", type=int, help="Số keyframe tối đa đi tới (mặc định theo config)")
    pr.add_argument("--detectors", nargs="*", help="Detector đã chạy cho workspace (mặc định theo config)")
    pr.set_defaults(func=cmd_propagate)

    ep = sub.add_parser("eval-propagation", help="Thí nghiệm keyframe hoàn hảo: lan truyền GT, so với GT")
    ep.add_argument("--scenes", nargs="*", help="Tên scene (mặc định: mọi scene có trong workspace)")
    ep.add_argument("--max-frames", type=int, help="Số keyframe tối đa đi tới")
    ep.add_argument("--detectors", nargs="*", help="Detector đã chạy cho workspace (mặc định theo config)")
    ep.add_argument("--out", default="eval/results")
    ep.set_defaults(func=cmd_eval_propagation)

    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
