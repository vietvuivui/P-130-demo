"""DAM4SAM (SAM 2.1 + bộ nhớ nhận biết vật gây nhiễu) cho lan truyền box 2D — so với tracker hiện tại (flow + ByteTrack
/ BoT-SORT) bằng thí nghiệm keyframe hoàn hảo (src/services/sequence.py: evaluate_propagation) và HOTA/MOTA/IDF1.

Cần GPU (RTX 3090: ~0.1–0.2 s/ảnh/vật với hiera-large ở 1600x900; keyframe 2 fps + sweep 12 Hz => ~6 ảnh mỗi keyframe).

Cài một lần (trong môi trường có torch CUDA, ví dụ .venv của requirements-ml.txt):

    git clone https://github.com/jovanavidenovic/DAM4SAM.git tools2d/DAM4SAM
    cd tools2d/DAM4SAM
    pip install -e .                      # cài SAM 2 (có CUDA kernel; lỗi thì: python setup.py build_ext --inplace)
    pip install vot-toolkit==0.7.1 vot-trax==4.0.2   # DAM4SAM import `vot` (không nằm trong pip install -e .)
    # checkpoint (~900 MB) vào tools2d/DAM4SAM/checkpoints/ — Windows PowerShell:
    #   curl.exe -L -o checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    python tools2d/dam4sam.py --check     # kiểm tra cài đặt (import, checkpoint, CUDA) mà không chạy gì

Chạy (workspace đã `python -m src.cli run` + `detect-sweeps`, cache detection có sẵn):

    python tools2d/dam4sam.py --dataroot ..\\v1.0-trainval --workspace data\\eval_temporal\\ws_dev --scenes scene-0035
    scripts\\tasks.ps1 dam4sam -Workspace data\\eval_temporal\\ws_dev -Scenes scene-0035,scene-0097   # PowerShell: dấu phẩy

Cấu hình đã có kết quả trong <out>/dam4sam.json (cùng workspace + scenes) được bỏ qua; --force để chạy lại.

In bảng: nhãn đúng / box sai / đổi ID / mất dấu, HOTA, MOTA, IDF1 và thời gian cho từng cấu hình:
    flow+byte (mặc định), flow+botsort, dam4sam+byte, dam4sam+botsort.
Chỉ chọn cấu hình trên dev (3 scene demo); báo trên held-out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "flow+byte": ("always", "byte"),
    "flow+botsort": ("always", "botsort"),
    "dam4sam+byte": ("dam4sam", "byte"),
    "dam4sam+botsort": ("dam4sam", "botsort"),
    "dam4sam+none": ("dam4sam", "none"),  # DAM4SAM thuần: không ghép với detection, box = hộp bao mask
}
# Thứ tự từ đơn giản / rẻ tới phức tạp / đắt: khi hai cấu hình ngang nhau (chênh HOTA < TIE) thì chọn cái đứng trước
COST_ORDER = ["flow+byte", "flow+botsort", "dam4sam+none", "dam4sam+byte", "dam4sam+botsort"]
TIE = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot")
    ap.add_argument("--check", action="store_true", help="Chỉ kiểm tra cài đặt DAM4SAM rồi thoát")
    ap.add_argument("--force", action="store_true", help="Chạy lại cả cấu hình đã có kết quả")
    ap.add_argument("--allow-sparse", action="store_true", help="Vẫn chạy khi cache detection thiếu ảnh 12 Hz")
    ap.add_argument("--no-detect", action="store_true", help="Không tự detect ảnh 12 Hz còn thiếu trong cache")
    ap.add_argument("--bench", action="store_true", help="Đo tăng tốc DAM4SAM (dùng chung encoder + autocast) trước / sau")
    ap.add_argument("--slow", action="store_true", help="Tắt tăng tốc DAM4SAM (chạy đúng như repo gốc)")
    ap.add_argument("--version", default=os.environ.get("NUSCENES_VERSION", "v1.0-trainval"))
    ap.add_argument("--workspace", default=os.environ.get("WORKSPACE_DIR", "data/workspace"))
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--model", default="sam21pp-L", help="sam21pp-L | sam21pp-B | sam21pp-S | sam21pp-T")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--out", default="eval/results/dam4sam")
    args = ap.parse_args()

    need_sam = args.check or any(CONFIGS[c][0] == "dam4sam" for c in args.configs)
    if need_sam:
        from src.services.dam4sam import check_install

        problems = check_install(args.model)
        for p in problems:
            print("THIẾU:", p)
        if args.check:
            print("DAM4SAM sẵn sàng" if not problems else "DAM4SAM chưa sẵn sàng")
            raise SystemExit(1 if problems else 0)
        if problems:  # báo ngay, không để chạy xong các cấu hình flow rồi mới lỗi
            raise SystemExit("Cài xong các mục THIẾU ở trên rồi chạy lại (hướng dẫn ở đầu file này).")
    if not args.dataroot:
        ap.error("--dataroot là bắt buộc")

    from src.models.qa_config import load_autolabel_config
    from src.services import trackeval as te
    from src.services.detectors import DetectorEnsemble
    from src.services.nuscenes_data import NuScenesMini
    from src.services.sequence import NuScenesSequenceSource, evaluate_propagation
    from src.services.store import WorkspaceStore

    config = load_autolabel_config(ROOT / "configs" / "autolabel.yaml")
    config.propagation.dam4sam_model = args.model
    data = NuScenesMini(args.dataroot, args.version)
    ws = Path(args.workspace)
    store = WorkspaceStore(ws)
    from src.services import botsort

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_key = {"workspace": str(ws), "scenes": sorted(args.scenes or []), "max_frames": args.max_frames,
               "model": args.model, "fast": not args.slow}  # fmt: skip
    config.propagation.dam4sam_share_encoder = config.propagation.dam4sam_autocast = not args.slow
    # Mọi cấu hình đều đọc detection từ cache; cache trống (chưa `detect-sweeps`, hoặc khoá cache đổi vì đổi weights /
    # prompt / ngưỡng) thì tracker chỉ trôi theo flow, các cấu hình ra giống hệt nhau và không nói lên gì
    ensemble0 = DetectorEnsemble(config, ws / "cache" / "detections")
    source0 = NuScenesSequenceSource(data, ensemble0)
    scenes = args.scenes or sorted({f.scene for f in store.list_frames()})
    n_img = n_cached = 0
    for sc in scenes:
        for im in source0.timeline(sc, config.camera):
            n_img += 1
            n_cached += ensemble0.load_cached(im.sd_token) is not None
    key = ensemble0._cache_file(config.detection.detectors[0], "x").parent.name
    print(f"Cache detection ({key}): {n_cached}/{n_img} ảnh của {len(scenes)} scene có detection", flush=True)
    if n_cached < n_img and not args.no_detect:
        print(f"Detect {n_img - n_cached} ảnh còn thiếu (GPU)…", flush=True)
        for sc in scenes:
            todo = [im for im in source0.timeline(sc, config.camera) if ensemble0.load_cached(im.sd_token) is None]
            for i in range(0, len(todo), 16):
                ensemble0.detect_batch([(im.sd_token, data.dataroot / im.path) for im in todo[i : i + 16]])
            print(f"  {sc}: detect thêm {len(todo)} ảnh", flush=True)
        n_cached = sum(ensemble0.load_cached(im.sd_token) is not None
                       for sc in scenes for im in source0.timeline(sc, config.camera))  # fmt: skip
    if n_img and n_cached / n_img < 0.9 and not args.allow_sparse:
        raise SystemExit(
            "Thiếu detection ở ảnh 12 Hz: chạy trước\n"
            f"  $env:WORKSPACE_DIR='{ws}'; $env:NUSCENES_DATAROOT='{args.dataroot}'; $env:NUSCENES_VERSION='{args.version}'\n"
            f"  python -m src.cli detect-sweeps --scenes {' '.join(scenes)}\n"
            "(hoặc --allow-sparse để vẫn chạy; kết quả khi đó chỉ phản ánh flow, không phải tracker)"
        )

    if args.bench:
        bench(store, data, config, ws, scenes[0], out)

    results = {}
    prev = out / "dam4sam.json"
    if prev.is_file() and not args.force:
        old = json.loads(prev.read_text(encoding="utf-8"))
        if old.get("_run") == run_key:
            results = {k: v for k, v in old.items() if k in CONFIGS}
    for name in args.configs:
        if name in results:
            print(name, "(đã có, bỏ qua; --force để chạy lại)", results[name], flush=True)
            continue
        botsort.STATS.update(calls=0, changed=0, rescued=0)
        flow, assoc = CONFIGS[name]
        cfg = config.model_copy(deep=True)
        cfg.propagation.flow, cfg.propagation.association = flow, assoc
        source = NuScenesSequenceSource(data, DetectorEnsemble(cfg, ws / "cache" / "detections"))
        t0 = time.perf_counter()
        r = evaluate_propagation(store, source, cfg, args.scenes, max_frames=args.max_frames, start_every=5)
        rows = r["per_hop"]
        results[name] = {
            "seconds": round(time.perf_counter() - t0, 1),
            "outputs": sum(x["tracks_output"] for x in rows),
            "correct": sum(x["correct"] for x in rows),
            "wrong": sum(x["wrong"] for x in rows),
            "id_switch": sum(x["id_switch"] for x in rows),
            "lost": sum(x.get("lost", 0) for x in rows),
            **{k: r["trackeval"][k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")},
        }
        if assoc == "botsort":  # số lần ghép BoT-SORT khác với ghép theo IoU thuần (0 = ngoại hình không đổi gì)
            results[name]["botsort"] = dict(botsort.STATS)
        print(name, results[name], flush=True)
        (out / "dam4sam.json").write_text(json.dumps({"_run": run_key, **results}, indent=1, ensure_ascii=False), encoding="utf-8")
    # HOTA / MOTA / IDF1 của nhãn hiện có trong workspace (sau lần lan truyền cuối) — tham khảo
    results["trackeval_workspace"] = te.evaluate_workspace(store, "2d", scenes=args.scenes)["overall"]
    (out / "dam4sam.json").write_text(json.dumps({"_run": run_key, **results}, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    done = [n for n in COST_ORDER if n in results]
    for name in done:
        v = results[name]
        te_cols = " | ".join(f"{v[k]:.3f}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"))
        lines.append(f"| {name} | {v['correct']} | {v['wrong']} | {v['id_switch']} | {v['lost']} | {te_cols} | {v['seconds']} |")
    lines += ["", *recommend(results, done)]
    (out / "dam4sam.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def recommend(results: dict, done: list[str]) -> list[str]:
    """Chọn cấu hình: HOTA cao nhất; trong khoảng TIE của cao nhất thì lấy cái đơn giản / rẻ nhất (COST_ORDER)."""
    if not done:
        return []
    best = max(results[n]["HOTA"] for n in done)
    pick = next(n for n in done if results[n]["HOTA"] >= best - TIE)
    top = max(done, key=lambda n: results[n]["HOTA"])
    base = results.get("flow+byte")
    out = [f"**Khuyến nghị: `{pick}`** (HOTA {results[pick]['HOTA']:.3f}, IDF1 {results[pick]['IDF1']:.3f}, {results[pick]['seconds']} s)."]
    if pick != top:
        out.append(f"`{top}` có HOTA cao nhất ({results[top]['HOTA']:.3f}) nhưng chỉ hơn {results[top]['HOTA'] - results[pick]['HOTA']:.3f} "
                   f"(< {TIE}), không đáng thêm độ phức tạp / thời gian ({results[top]['seconds']} s).")  # fmt: skip
    if base and pick != "flow+byte":
        v = results[pick]
        out.append(f"So với mặc định `flow+byte`: HOTA {base['HOTA']:.3f} → {v['HOTA']:.3f}, IDF1 {base['IDF1']:.3f} → {v['IDF1']:.3f}, "
                   f"nhãn đúng {base['correct']} → {v['correct']}, thời gian ×{v['seconds'] / max(base['seconds'], 0.1):.0f}.")  # fmt: skip
    if "dam4sam+byte" in results and "dam4sam+none" in results:
        a, b = results["dam4sam+none"], results["dam4sam+byte"]
        out.append(f"Ghép DAM4SAM với detection (byte) so với DAM4SAM thuần: HOTA {a['HOTA']:.3f} → {b['HOTA']:.3f}, "
                   f"box sai {a['wrong']} → {b['wrong']}, mất dấu {a['lost']} → {b['lost']}.")  # fmt: skip
    if "dam4sam+byte" in results and "dam4sam+botsort" in results:
        a, b = results["dam4sam+byte"], results["dam4sam+botsort"]
        st = b.get("botsort", {})
        out.append(f"Thêm BoT-SORT vào DAM4SAM: HOTA {a['HOTA']:.3f} → {b['HOTA']:.3f}, đổi ID {a['id_switch']} → {b['id_switch']} "
                   f"(ngoại hình đổi {st.get('changed', '?')}/{st.get('calls', '?')} lần ghép).")  # fmt: skip
    out.append(f"Quy tắc chọn: HOTA cao nhất; chênh dưới {TIE} thì lấy cấu hình đơn giản / nhanh hơn. Chỉ chọn trên dev.")
    return out


def bench(store, data, config, ws: Path, scene: str, out: Path) -> None:
    """Đo tăng tốc DAM4SAM trên một đoạn ngắn (keyframe đầu của scene, đi 2 keyframe): gốc vs dùng chung encoder +
    autocast; kiểm tra kết quả có đổi không."""
    from src.services import dam4sam
    from src.services.detectors import DetectorEnsemble
    from src.services.sequence import NuScenesSequenceSource, evaluate_propagation

    rows = {}
    for name, fast in (("gốc (mỗi vật mã hoá lại ảnh, float32)", False), ("dùng chung encoder + autocast", True)):
        cfg = config.model_copy(deep=True)
        cfg.propagation.flow, cfg.propagation.association = "dam4sam", "byte"
        cfg.propagation.dam4sam_share_encoder = cfg.propagation.dam4sam_autocast = fast
        source = NuScenesSequenceSource(data, DetectorEnsemble(cfg, ws / "cache" / "detections"))
        for e in dam4sam._ENCODERS.values():
            e.hits = e.misses = 0
        t0 = time.perf_counter()
        r = evaluate_propagation(store, source, cfg, [scene], max_frames=2)
        dt = time.perf_counter() - t0
        enc = next(iter(dam4sam._ENCODERS.values()), None)
        rows[name] = {"seconds": round(dt, 1), "correct": sum(x["correct"] for x in r["per_hop"]),
                      "wrong": sum(x["wrong"] for x in r["per_hop"]), "HOTA": r["trackeval"]["HOTA"],
                      "tracks": r["tracks_started"], "encoder_runs": enc.misses if enc else None,
                      "encoder_reused": enc.hits if enc else None}  # fmt: skip
        print("bench", name, rows[name], flush=True)
    a, b = rows.values()
    print(f"Tăng tốc DAM4SAM: {a['seconds']} s → {b['seconds']} s (×{a['seconds'] / max(b['seconds'], 0.1):.1f}); "
          f"nhãn đúng {a['correct']} → {b['correct']}, HOTA {a['HOTA']:.3f} → {b['HOTA']:.3f}", flush=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
