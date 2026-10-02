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
}


def check_install(model: str = "sam21pp-L") -> list[str]:
    """Những thứ còn thiếu để chạy DAM4SAM (rỗng = đủ). Không nạp mô hình."""
    import importlib.util

    problems = []
    repo = Path(os.environ.get("DAM4SAM_DIR") or ROOT / "tools2d" / "DAM4SAM")
    if not (repo / "dam4sam_tracker.py").is_file():
        return [f"repo DAM4SAM ở {repo}: git clone https://github.com/jovanavidenovic/DAM4SAM.git tools2d/DAM4SAM"]
    for mod, hint in (("vot", "pip install vot-toolkit==0.7.1 vot-trax==4.0.2"),
                      ("sam2", f"cd {repo} rồi pip install -e ."),
                      ("torch", "pip install torch (bản CUDA)"), ("yaml", "pip install pyyaml")):  # fmt: skip
        if importlib.util.find_spec(mod) is None:
            problems.append(f"gói `{mod}`: {hint}")
    ckpt = {"sam21pp-L": "sam2.1_hiera_large.pt", "sam21pp-B": "sam2.1_hiera_base_plus.pt"}.get(model)
    if ckpt and not (repo / "checkpoints" / ckpt).is_file():
        problems.append(
            f"checkpoint {repo / 'checkpoints' / ckpt}: curl.exe -L -o {repo / 'checkpoints' / ckpt} "
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt}"
        )
    if importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            problems.append("CUDA: torch không thấy GPU (DAM4SAM nạp mô hình lên cuda:0)")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot")
    ap.add_argument("--check", action="store_true", help="Chỉ kiểm tra cài đặt DAM4SAM rồi thoát")
    ap.add_argument("--force", action="store_true", help="Chạy lại cả cấu hình đã có kết quả")
    ap.add_argument("--allow-sparse", action="store_true", help="Vẫn chạy khi cache detection thiếu ảnh 12 Hz")
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
    run_key = {"workspace": str(ws), "scenes": sorted(args.scenes or []), "max_frames": args.max_frames, "model": args.model}
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
    if n_img and n_cached / n_img < 0.9 and not args.allow_sparse:
        raise SystemExit(
            "Thiếu detection ở ảnh 12 Hz: chạy trước\n"
            f"  $env:WORKSPACE_DIR='{ws}'; $env:NUSCENES_DATAROOT='{args.dataroot}'; $env:NUSCENES_VERSION='{args.version}'\n"
            f"  python -m src.cli detect-sweeps --scenes {' '.join(scenes)}\n"
            "(hoặc --allow-sparse để vẫn chạy; kết quả khi đó chỉ phản ánh flow, không phải tracker)"
        )

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
        }
        if assoc == "botsort":  # số lần ghép BoT-SORT khác với ghép theo IoU thuần (0 = ngoại hình không đổi gì)
            results[name]["botsort"] = dict(botsort.STATS)
        print(name, results[name], flush=True)
        (out / "dam4sam.json").write_text(json.dumps({"_run": run_key, **results}, indent=1, ensure_ascii=False), encoding="utf-8")
    # HOTA / MOTA / IDF1 của nhãn hiện có trong workspace (sau lần lan truyền cuối) — tham khảo
    results["trackeval_workspace"] = te.evaluate_workspace(store, "2d", scenes=args.scenes)["overall"]
    (out / "dam4sam.json").write_text(json.dumps({"_run": run_key, **results}, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | Giây |", "|---|---|---|---|---|---|"]
    for name in args.configs:
        v = results[name]
        lines.append(f"| {name} | {v['correct']} | {v['wrong']} | {v['id_switch']} | {v['lost']} | {v['seconds']} |")
    (out / "dam4sam.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
