"""So sánh trước / sau optical flow trên một workspace có GT (nút "Chạy đánh giá" ở tab ⚙ Cài đặt, và
tools2d/eval_temporal.py). Mọi cấu hình dùng chung cache detection của workspace, chỉ khác phần xử lý sau detector.

2D: base | flow | flow+mean | flow+linked | mean (xem tools2d/eval_temporal.py). Lan truyền: thí nghiệm keyframe hoàn
hảo (GT keyframe 0, 5, 10… làm nhãn người, tối đa 10 keyframe), propagation.flow = off | missing | always.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

VARIANTS = {
    "base": dict(flow=False, rescore="off"),
    "flow": dict(flow=True, rescore="off"),
    "flow+mean": dict(flow=True, rescore="mean"),
    "flow+linked": dict(flow=True, rescore="linked"),
    "mean": dict(flow=False, rescore="mean"),
    "low": dict(flow=False, rescore="off", sweep_min_score=0.1),
    "low+mean": dict(flow=False, rescore="mean", sweep_min_score=0.1),
    "low+flow+mean": dict(flow=True, rescore="mean", sweep_min_score=0.1),
}
# Mặc định chỉ chạy 5 cấu hình chính (nút trên web); "low*" = nhận cả box sweep score ≥ 0.1 làm bằng chứng (kiểu
# ByteTrack) — đã đo, làm lỗi lọt qua tăng (report.md mục 5), chạy thêm bằng --variants nếu cần.
DEFAULT_VARIANTS = ["base", "flow", "flow+mean", "flow+linked", "mean"]
# tên -> (propagation.flow, propagation.association). "off" = cách làm trước khi có flow / ByteTrack
PROP_MODES = {
    "off": ("off", "single"),
    "missing": ("missing", "single"),
    "always": ("always", "single"),
    "always+byte": ("always", "byte"),
    "always+botsort": ("always", "botsort"),
    "off+botsort": ("off", "botsort"),  # BoT-SORT thuần: GMC thay flow + ngoại hình
}


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
    """Auto-label lại đúng các keyframe nuScenes có trong workspace gốc (không detect thêm frame mới)."""
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
    cfg.qa.temporal.sweep_min_score = opts.get("sweep_min_score")
    store = WorkspaceStore(out_ws)
    keys = sorted(
        (f.scene, f.index, f.sample_token)
        for f in WorkspaceStore(base_ws).list_frames()
        if (scenes is None or f.scene in scenes) and f.sample_token in data.sample
    )
    t0 = time.perf_counter()
    pipe = AutoLabelPipeline(data, store, cfg)
    pipe.run_id = f"eval-{name}"
    for scene, index, token in keys:
        pipe.process(scene, index, token, overwrite=True)
    print(f"  {name}: auto-label lại {len(keys)} keyframe, {time.perf_counter() - t0:.0f}s", flush=True)
    return metrics_2d(store)


def run_propagation(data, ws: Path, config, scenes=None) -> dict:
    from src.services.detectors import DetectorEnsemble
    from src.services.sequence import NuScenesSequenceSource, evaluate_propagation
    from src.services.store import WorkspaceStore

    store = WorkspaceStore(ws)
    out = {}
    for mode, (flow, assoc) in PROP_MODES.items():
        cfg = config.model_copy(deep=True)
        cfg.propagation.flow, cfg.propagation.association = flow, assoc
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


def gt_frames(store) -> int:
    return sum(store.load_aux("gt", f.frame_id) is not None for f in store.list_frames())


def evaluate(data, base_ws: Path, out_dir: Path, config, variants=None, scenes=None,
             progress: Callable[[float, str], None] = lambda p, m: None) -> dict:  # fmt: skip
    """Chạy mọi cấu hình 2D + lan truyền; ghi out_dir/temporal_eval.json; xoá workspace tạm."""
    from src.services.store import WorkspaceStore

    variants = variants or DEFAULT_VARIANTS
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = len(variants) + 1
    result = {"workspace": str(base_ws), "as_run": metrics_2d(WorkspaceStore(base_ws)), "variants": {},
              "started_at": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    try:
        for i, name in enumerate(variants):
            progress(i / steps, f"2D: {name} ({i + 1}/{len(variants)})")
            result["variants"][name] = run_variant(name, VARIANTS[name], data, base_ws, out_dir / f"ws_{name}", config,
                                                   scenes)  # fmt: skip
        progress(len(variants) / steps, "Lan truyền nhãn: off / missing / always")
        if "base" not in variants:
            run_variant("base", VARIANTS["base"], data, base_ws, out_dir / "ws_base", config, scenes)
        result["propagation"] = run_propagation(data, out_dir / "ws_base", config, scenes)
    finally:
        for name in set(variants) | {"base"}:
            shutil.rmtree(out_dir / f"ws_{name}", ignore_errors=True)
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M")
    (out_dir / "temporal_eval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def summary_rows(result: dict) -> dict:
    """Số liệu gọn cho bảng (UI và terminal)."""
    rows2d = []
    for name, v in result.get("variants", {}).items():
        low = round(v["objects"] * (v["low_risk_share"] or 0))
        rows2d.append({"name": name, "mAP50": v["mAP50"], "f1": v["f1"], "review": v["objects"] - low,
                       "slip": round(low * (v["low_risk_error_rate"] or 0)), "fn": v["fn"], "fp": v["fp"],
                       "s_per_frame": v["s_per_frame"], "frames": v["frames"]})  # fmt: skip
    rows_prop = [{"mode": m, **{k: v[k] for k in ("correct", "outputs", "correct_rate", "id_switch", "lost", "mean_iou",
                                                  "seconds", "tracks_started")}}
                 for m, v in result.get("propagation", {}).items()]  # fmt: skip
    return {"rows2d": rows2d, "propagation": rows_prop}


def print_summary(result: dict) -> None:
    """Bảng ngắn: 2D (mAP, việc của người) và lan truyền, để đọc ngay trên terminal."""
    print("\n== 2D: cùng detection, khác xử lý sau detector ==")
    print(
        f"{'cấu hình':<12} {'mAP50':>6} {'F1':>6} {'xem tay':>8} {'lọt lô':>7} {'vẽ thêm':>8} {'box sai':>8} {'s/frame':>8}"
    )
    rows = summary_rows(result)
    for r in rows["rows2d"]:
        print(f"{r['name']:<12} {r['mAP50'] or 0:>6.3f} {r['f1'] or 0:>6.3f} {r['review']:>8} {r['slip']:>7} "
              f"{r['fn']:>8} {r['fp']:>8} {r['s_per_frame'] or 0:>8.2f}")  # fmt: skip
    print("xem tay = box medium/high; lọt lô = box sai nằm trong nhóm low (duyệt theo lô); vẽ thêm = GT bị sót")
    print("\n== Lan truyền nhãn (GT keyframe 0, 5, 10… làm nhãn người; tối đa 10 keyframe) ==")
    print(f"{'flow':<8} {'đúng':>6} {'box ra':>7} {'tỉ lệ':>6} {'đổi ID':>7} {'mất dấu':>8} {'IoU TB':>7} {'giây':>6}")
    for v in rows["propagation"]:
        print(f"{v['mode']:<8} {v['correct']:>6} {v['outputs']:>7} {v['correct_rate'] or 0:>6.3f} {v['id_switch']:>7} "
              f"{v['lost']:>8} {v['mean_iou'] or 0:>7.3f} {v['seconds']:>6.1f}")  # fmt: skip
