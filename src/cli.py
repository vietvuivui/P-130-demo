"""CLI chạy batch.

python -m src.cli run --limit 20                       # auto-label + QA Agent cho 20 keyframe đầu
python -m src.cli run --scenes scene-0061 scene-0103   # theo scene
python -m src.cli run --detectors yoloe yolo26 --overwrite
python -m src.cli evaluate                             # mAP + flag recall/precision -> eval/results/

Lan truyền nhãn trên video:
python -m src.cli detect-sweeps --scenes scene-0061    # detect mọi ảnh CAM_FRONT 12Hz của scene (GPU, có cache)
python -m src.cli propagate scene-0061_000             # lan truyền từ frame đã approve sang các keyframe sau
python -m src.cli eval-propagation                     # thí nghiệm keyframe hoàn hảo -> eval/results/

QC:
python -m src.cli qc-report                            # checklist sẵn sàng phát hành; exit 1 nếu chưa READY
python -m src.cli quick-check vendor_labels.json       # kiểm nhanh file nhãn ngoài; exit 1 nếu có lỗi (error)
Phần 3D (dự đoán từ tools3d/run3d.py trên máy có GPU):
python -m src.cli label3d --model pointpillars          # kiểm chứng box 3D bằng camera, tạo frame 3D để duyệt
python -m src.cli evaluate3d --model pointpillars       # kết luận kiểm chứng so với nhãn gốc -> eval/results/det3d/

python -m src.cli profile --limit 10                   # một frame tốn thời gian ở bước nào (không dùng cache)
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


def cmd_qc_report(args) -> None:
    from src.services.qc import dataset_report
    from src.services.store import WorkspaceStore

    settings = get_settings()
    report = dataset_report(WorkspaceStore(settings.workspace_dir), load_autolabel_config(settings.autolabel_config))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"{report['status']} — {report['frames']['approved']}/{report['frames']['total']} frame đã approve")
        for c in report["checks"]:
            print(f"  [{'x' if c['ok'] else ' '}] {c['label']}: {c['detail']}")
    raise SystemExit(0 if report["status"] == "READY" else 1)


def cmd_quick_check(args) -> None:
    from src.services.qc import QCError, quick_check
    from src.services.store import WorkspaceStore

    settings = get_settings()
    path = Path(args.file)
    try:
        res = quick_check(
            WorkspaceStore(settings.workspace_dir),
            load_autolabel_config(settings.autolabel_config),
            path.name,
            path.read_bytes(),
        )
    except QCError as e:
        raise SystemExit(f"Không đọc được {path}: {e}") from e
    if args.json:
        print(res.model_dump_json(indent=2))
    else:
        print(
            f"{res.n_frames} frame, {res.n_labels} nhãn: {res.n_errors} lỗi, {res.n_warnings} cảnh báo, "
            f"{res.n_acked} cảnh báo người đã xác nhận ({res.elapsed_ms} ms) {res.by_code}"
        )
        for msg in res.parse_errors:
            print(f"  ! {msg}")
        for fr in res.frames:
            for f in (f for f in fr.findings if not f.acked):
                who = f"#{f.object_id}" if f.object_id else "-"
                print(f"  {f.severity:7s} {fr.frame_id or fr.file_name} {who} {f.code}: {f.message}")
    raise SystemExit(1 if res.n_errors or res.parse_errors else 0)
def _preds_file(model: str, path: str | None) -> Path:
    p = Path(path) if path else Path("eval/results/det3d") / model / "ui_preds.json"
    if not p.is_file():
        raise SystemExit(f"Không thấy {p}: chạy tools3d/run3d.py trên máy có GPU rồi chép eval/results/det3d/ về")
    return p


def cmd_label3d(args) -> None:
    from src.services.label3d import run_label3d
    from src.services.nuscenes_data import NuScenesMini
    from src.services.store import WorkspaceStore

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    preds = json.loads(_preds_file(args.model, args.preds).read_text())["results"]
    min_score = _min_score3d(args.model, args.min_score, config.verify3d.min_score)
    data = NuScenesMini(settings.nuscenes_dataroot, settings.nuscenes_version)
    n = run_label3d(
        WorkspaceStore(settings.workspace_dir), data, config, args.model, preds, args.scenes, args.overwrite, min_score
    )
    print(f"Xong {n} frame 3D ({args.model}, ngưỡng {min_score}) -> {settings.workspace_dir}/frames3d/{args.model}")


def _min_score3d(model: str, arg: str, default: float) -> float:
    """--min-score số cụ thể, hoặc "auto": ngưỡng F1 cao nhất trong eval/results/det3d/<model>/metrics.json."""
    from src.services.label3d import auto_min_score

    return float(arg) if arg != "auto" else auto_min_score(model, default)


def cmd_evaluate3d(args) -> None:
    from src.services.eval3d import evaluate_verification
    from src.services.store import WorkspaceStore

    store = WorkspaceStore(get_settings().workspace_dir)
    for model in args.model or store.models3d():
        res = evaluate_verification(store, model)
        out = Path(args.out) / model
        out.mkdir(parents=True, exist_ok=True)
        (out / "verify_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(
            f"{model}: {res['n_boxes']} box, tự duyệt {res['auto_share']:.0%} (đúng {res['auto_precision']:.1%}), "
            f"bắt {res['error_recall']:.0%} box sai"
        )


def cmd_trackeval(args) -> None:
    """HOTA / MOTA / IDF1 của nhãn lan truyền 2D và box 3D (src/services/trackeval.py); tuỳ chọn chạy TrackEval chính thức."""
    from src.services import trackeval as te
    from src.services.store import WorkspaceStore

    store = WorkspaceStore(get_settings().workspace_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.mode in ("2d", "both"):
        results["2d"] = te.evaluate_workspace(store, "2d", scenes=args.scenes, min_score=args.min_score,
                                              propagated_only=args.propagated_only)  # fmt: skip
    if args.mode in ("3d", "both"):
        for model in args.model or store.models3d():
            results[f"3d:{model}"] = te.evaluate_workspace(store, "3d", model, scenes=args.scenes)
    lines = ["| Chế độ | Frame | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o = r["overall"]
        lines.append(f"| {name} | {o['frames']} | {o['HOTA']:.3f} | {o['DetA']:.3f} | {o['AssA']:.3f} | {o['MOTA']:.3f} | "
                     f"{o['IDF1']:.3f} | {o['IDSW']} | {o['FP']} | {o['FN']} |")  # fmt: skip
        if args.export_mot:
            mode = "2d" if name == "2d" else "3d"
            seqs = {s: (te.sequence_2d(store, s, args.min_score or 0.0, args.propagated_only) if mode == "2d"
                        else te.sequence_3d(store, name.split(":", 1)[1], s)) for s in r["scenes"]}  # fmt: skip
            mot = te.export_mot(seqs, out / "mot" / name.replace(":", "_"), "autolabel", mode)
            official = te.run_trackeval(mot, "autolabel")
            r["official"] = official
            if official:
                lines.append(f"| ↳ TrackEval chính thức | | {official['HOTA']:.3f} | {official['DetA']:.3f} | "
                             f"{official['AssA']:.3f} | {official['MOTA']:.3f} | {official['IDF1']:.3f} | {official['IDSW']} | | |")  # fmt: skip
            else:
                lines.append(f"| ↳ TrackEval chưa cài (pip install git+https://github.com/JonathonLuiten/TrackEval.git); đã ghi MOTChallenge ở {mot} |")
    (out / "trackeval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    report = "\n".join(lines) + "\n"
    (out / "trackeval.md").write_text(report, encoding="utf-8")
    print(report)


def cmd_profile(args) -> None:
    from src.services.nuscenes_data import NuScenesMini
    from src.services.profiling import print_profile, profile

    settings = get_settings()
    config = load_autolabel_config(settings.autolabel_config)
    data = NuScenesMini(settings.nuscenes_dataroot, settings.nuscenes_version)
    print_profile(profile(data, config, args.scenes, args.limit, privacy=not args.no_privacy))


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

    l3 = sub.add_parser("label3d", help="Kiểm chứng dự đoán của mô hình 3D bằng camera, tạo frame 3D để duyệt")
    l3.add_argument("--model", required=True, help="Tên mô hình, ví dụ pointpillars (thư mục eval/results/det3d/)")
    l3.add_argument("--preds", help="File dự đoán chuẩn nuScenes (mặc định eval/results/det3d/<model>/ui_preds.json)")
    l3.add_argument("--scenes", nargs="*", help="Tên scene (mặc định: mọi scene có dự đoán)")
    l3.add_argument("--overwrite", action="store_true")
    l3.add_argument(
        "--min-score", default="auto", help="Ngưỡng điểm giữ box; auto = ngưỡng F1 cao nhất của mô hình trên tập val"
    )
    l3.set_defaults(func=cmd_label3d)

    e3 = sub.add_parser("evaluate3d", help="Kết luận kiểm chứng 3D so với nhãn gốc")
    e3.add_argument("--model", nargs="*", help="Mặc định: mọi mô hình có frame 3D")
    e3.add_argument("--out", default="eval/results/det3d")
    e3.set_defaults(func=cmd_evaluate3d)

    pf = sub.add_parser("profile", help="Đo thời gian từng bước cho N keyframe, không dùng cache")
    pf.add_argument("--scenes", nargs="*", help="Tên scene (mặc định: scene đầu tiên có dữ liệu)")
    pf.add_argument("--limit", type=int, default=10, help="Số keyframe")
    pf.add_argument("--no-privacy", action="store_true", help="Bỏ đo làm mờ mặt / biển số")
    pf.set_defaults(func=cmd_profile)

    ep = sub.add_parser("eval-propagation", help="Thí nghiệm keyframe hoàn hảo: lan truyền GT, so với GT")
    ep.add_argument("--scenes", nargs="*", help="Tên scene (mặc định: mọi scene có trong workspace)")
    ep.add_argument("--max-frames", type=int, help="Số keyframe tối đa đi tới")
    ep.add_argument("--detectors", nargs="*", help="Detector đã chạy cho workspace (mặc định theo config)")
    ep.add_argument("--out", default="eval/results")
    ep.set_defaults(func=cmd_eval_propagation)

    tk = sub.add_parser("trackeval", help="HOTA / MOTA / IDF1 của nhãn lan truyền 2D và box 3D theo chuẩn TrackEval")
    tk.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    tk.add_argument("--model", nargs="*", help="Mô hình 3D (mặc định: mọi mô hình trong workspace)")
    tk.add_argument("--scenes", nargs="*")
    tk.add_argument("--min-score", type=float, help="Chỉ chấm box có score >= (2D mặc định 0, 3D 0.3)")
    tk.add_argument("--propagated-only", action="store_true", help="2D: chỉ chấm frame nhận nhãn lan truyền")
    tk.add_argument("--export-mot", action="store_true", help="Ghi MOTChallenge và chạy TrackEval chính thức nếu đã cài")
    tk.add_argument("--out", default="eval/results/trackeval")
    tk.set_defaults(func=cmd_trackeval)

    qr = sub.add_parser("qc-report", help="Checklist QC sẵn sàng phát hành (exit 1 nếu chưa READY)")
    qr.add_argument("--json", action="store_true", help="In toàn bộ báo cáo dạng JSON")
    qr.set_defaults(func=cmd_qc_report)

    qk = sub.add_parser("quick-check", help="Kiểm nhanh file nhãn COCO / JSONL (exit 1 nếu có lỗi)")
    qk.add_argument("file", help="File .json (COCO hoặc {'frames': [...]}) hoặc .jsonl")
    qk.add_argument("--json", action="store_true", help="In kết quả dạng JSON")
    qk.set_defaults(func=cmd_quick_check)

    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
