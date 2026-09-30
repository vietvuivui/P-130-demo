"""So sánh trước / sau khi dùng optical flow cho QA temporal, tính lại score theo sweep (idea 2) và lan truyền (idea 1).

Chạy trên một workspace đã `python -m src.cli run` (cache detection của keyframe + sweep t-2..t+2 đã có), nên mọi cấu hình
dùng đúng cùng một detection — chỉ khác phần xử lý sau detector:

    python tools2d/eval_temporal.py --dataroot <nuscenes> --workspace <ws đã run> --out eval/results/temporal/dev

Cấu hình 2D (mỗi cái auto-label lại toàn bộ keyframe vào một workspace riêng, chung cache detection):
    base        : như trước (so khớp sweep trực tiếp, score detector)
    flow        : so khớp sweep sau khi dời bằng optical flow (FLICKER / RECOVERED chính xác hơn), score detector
    flow+mean   : + score tính lại = trung bình keyframe và các sweep (sweep không thấy góp 0)
    flow+linked : + score tính lại = trung bình các lần thấy (kiểu Seq-NMS)
    mean        : score tính lại nhưng không flow (để xem flow có cần không)
Lan truyền (thí nghiệm keyframe hoàn hảo, src/services/sequence.py; bắt đầu ở keyframe 0, 5, 10… của mỗi scene, đi tối
đa 10 keyframe = 5 s): propagation.flow = off | missing | always.

Chọn cấu hình bằng scene dev (mAP50, rồi F1), báo kết quả chính trên held-out — không chỉnh gì theo held-out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "base": dict(flow=False, rescore="off"),
    "flow": dict(flow=True, rescore="off"),
    "flow+mean": dict(flow=True, rescore="mean"),
    "flow+linked": dict(flow=True, rescore="linked"),
    "mean": dict(flow=False, rescore="mean"),
}
PROP_MODES = ("off", "missing", "always")


def metrics_2d(store) -> dict:
    from src.services.evaluation import evaluate_map, evaluate_pr, evaluate_qa

    gt = {f.frame_id: g for f in store.list_frames() if (g := store.load_aux("gt", f.frame_id)) is not None}
    frames = [f for f in store.list_frames() if f.frame_id in gt]
    qa = evaluate_qa(frames, gt, 0.5)
    pr = evaluate_pr(frames, gt, 0.5)
    times = [f.autolabel_s for f in frames if f.autolabel_s is not None]
    m50 = evaluate_map(frames, gt, 0.5)
    m70 = evaluate_map(frames, gt, 0.7)
    return {
        "frames": len(frames),
        "mAP50": m50["map"],
        "mAP70": m70["map"],
        "ap50": {k: v["ap"] for k, v in m50["per_class"].items()},
        **{k: pr[k] for k in ("tp", "fp", "fn", "n_gt", "precision", "recall", "f1")},
        "objects": qa["objects_evaluated"],
        "need_fix": qa["objects_need_fix"],
        "low_risk_share": qa["low_risk_share"],
        "low_risk_error_rate": qa["low_risk_error_rate"],
        "flag_recall": qa["flag_recall"],
        "flag_precision": qa["flag_precision"],
        "flicker": qa["by_issue"].get("FLICKER"),
        "fn_total": qa["fn_total"],
        "fn_recovered": qa["fn_recovered_by_track"],
        "track_proposals": qa["track_proposals"],
        "track_precision": qa["track_proposal_precision"],
        "s_per_frame": round(float(np.mean(times)), 3) if times else None,
    }


def run_variant(name: str, opts: dict, data, base_ws: Path, out_ws: Path, config, scenes=None) -> dict:
    from src.services.pipeline import AutoLabelPipeline
    from src.services.store import WorkspaceStore

    if out_ws.exists():
        shutil.rmtree(out_ws)
    (out_ws / "cache").mkdir(parents=True)
    src_cache = (base_ws / "cache" / "detections").resolve()
    try:
        (out_ws / "cache" / "detections").symlink_to(src_cache, target_is_directory=True)
    except OSError:  # Windows không bật Developer Mode thì không tạo được symlink: chép cache (file JSON nhỏ)
        shutil.copytree(src_cache, out_ws / "cache" / "detections")
    cfg = config.model_copy(deep=True)
    cfg.qa.temporal.flow = opts["flow"]
    cfg.qa.temporal.rescore = opts["rescore"]
    store = WorkspaceStore(out_ws)
    t0 = time.perf_counter()
    AutoLabelPipeline(data, store, cfg).run(scenes=scenes)
    print(f"  {name}: auto-label lại {time.perf_counter() - t0:.0f}s", flush=True)
    return metrics_2d(store)


def run_propagation(data, ws: Path, config, scenes=None) -> dict:
    from src.services.detectors import DetectorEnsemble
    from src.services.sequence import NuScenesSequenceSource, evaluate_propagation
    from src.services.store import WorkspaceStore

    store = WorkspaceStore(ws)
    out = {}
    for mode in PROP_MODES:
        cfg = config.model_copy(deep=True)
        cfg.propagation.flow = mode
        source = NuScenesSequenceSource(data, DetectorEnsemble(cfg, ws / "cache" / "detections"))
        t0 = time.perf_counter()
        r = evaluate_propagation(store, source, cfg, scenes, max_frames=10, start_every=5)
        rows = r["per_hop"]
        n = sum(x["tracks_output"] for x in rows)
        out[mode] = {
            "seconds": round(time.perf_counter() - t0, 1),
            "starts": r["starts"],
            "tracks_started": r["tracks_started"],
            "outputs": n,
            "correct": sum(x["correct"] for x in rows),
            "id_switch": sum(x["id_switch"] for x in rows),
            "wrong": sum(x["wrong"] for x in rows),
            "lost": sum(x["lost"] for x in rows),
            "correct_rate": round(sum(x["correct"] for x in rows) / n, 4) if n else None,
            "mean_iou": round(sum(x["mean_iou"] * x["tracks_output"] for x in rows if x["mean_iou"]) / n, 4)
            if n
            else None,
            "per_hop": rows[:10],
            "calibration": r["calibration"],
        }
        print(f"  lan truyền flow={mode}: đúng {out[mode]['correct']}/{n}, IoU TB {out[mode]['mean_iou']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument(
        "--workspace", required=True, type=Path, help="workspace đã chạy `src.cli run` (có cache detection)"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--scenes", nargs="*", help="chỉ các scene này (mặc định: mọi scene của workspace)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "autolabel.yaml"))
    args = ap.parse_args()

    from src.models.qa_config import load_autolabel_config
    from src.services.nuscenes_data import NuScenesMini
    from src.services.store import WorkspaceStore

    config = load_autolabel_config(args.config)
    data = NuScenesMini(args.dataroot, args.version)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"workspace": str(args.workspace), "as_run": metrics_2d(WorkspaceStore(args.workspace)), "variants": {}}
    for name in args.variants:
        result["variants"][name] = run_variant(
            name, VARIANTS[name], data, args.workspace, args.out / f"ws_{name}", config, args.scenes
        )
        (args.out / "temporal_eval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    result["propagation"] = run_propagation(data, args.out / "ws_base", config, args.scenes)
    (args.out / "temporal_eval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in args.variants:
        shutil.rmtree(args.out / f"ws_{name}", ignore_errors=True)
    print_summary(result)


def print_summary(result: dict) -> None:
    """Bảng ngắn: 2D (mAP, việc của người) và lan truyền, để đọc ngay trên terminal."""
    print("\n== 2D: cùng detection, khác xử lý sau detector ==")
    print(f"{'cấu hình':<12} {'mAP50':>6} {'F1':>6} {'xem tay':>8} {'lọt lô':>7} {'vẽ thêm':>8} {'box sai':>8} {'s/frame':>8}")
    for name, v in result["variants"].items():
        low = round(v["objects"] * (v["low_risk_share"] or 0))
        slip = round(low * (v["low_risk_error_rate"] or 0))
        print(f"{name:<12} {v['mAP50']:>6.3f} {v['f1'] or 0:>6.3f} {v['objects'] - low:>8} {slip:>7} {v['fn']:>8} "
              f"{v['fp']:>8} {v['s_per_frame'] or 0:>8.2f}")  # fmt: skip
    print("xem tay = box medium/high; lọt lô = box sai nằm trong nhóm low (duyệt theo lô); vẽ thêm = GT bị sót")
    print("\n== Lan truyền nhãn (GT keyframe 0, 5, 10… làm nhãn người; tối đa 10 keyframe) ==")
    print(f"{'flow':<8} {'đúng':>6} {'box ra':>7} {'tỉ lệ':>6} {'đổi ID':>7} {'mất dấu':>8} {'IoU TB':>7} {'giây':>6}")
    for mode, v in result.get("propagation", {}).items():
        print(f"{mode:<8} {v['correct']:>6} {v['outputs']:>7} {v['correct_rate'] or 0:>6.3f} {v['id_switch']:>7} "
              f"{v['lost']:>8} {v['mean_iou'] or 0:>7.3f} {v['seconds']:>6.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
