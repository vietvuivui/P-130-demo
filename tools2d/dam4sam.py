"""DAM4SAM (SAM 2.1 + bộ nhớ nhận biết vật gây nhiễu) cho lan truyền box 2D — so với tracker hiện tại (flow + ByteTrack
/ BoT-SORT) bằng thí nghiệm keyframe hoàn hảo (src/services/sequence.py: evaluate_propagation) và HOTA/MOTA/IDF1.

Cần GPU (RTX 3090: ~0.1–0.2 s/ảnh/vật với hiera-large ở 1600x900; keyframe 2 fps + sweep 12 Hz => ~6 ảnh mỗi keyframe).

Cài một lần (trong môi trường có torch CUDA, ví dụ .venv của requirements-ml.txt):

    git clone https://github.com/jovanavidenovic/DAM4SAM.git tools2d/DAM4SAM
    cd tools2d/DAM4SAM
    pip install -e .                      # cài SAM 2 (có CUDA kernel; lỗi thì: python setup.py build_ext --inplace)
    pip install vot-toolkit-lite==0.2.0   # (nếu thiếu `vot` khi import)
    cd checkpoints && bash download_ckpts.sh   # tải sam2.1_hiera_*.pt (Windows: tải tay theo URL trong file)

Chạy (workspace đã `python -m src.cli run` + `detect-sweeps`, cache detection có sẵn):

    python tools2d/dam4sam.py --dataroot ..\\v1.0-trainval --workspace data\\eval_temporal\\ws_dev --scenes scene-0035
    scripts\\tasks.ps1 dam4sam -Workspace data\\eval_temporal\\ws_dev

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default=os.environ.get("NUSCENES_VERSION", "v1.0-trainval"))
    ap.add_argument("--workspace", default=os.environ.get("WORKSPACE_DIR", "data/workspace"))
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--model", default="sam21pp-L", help="sam21pp-L | sam21pp-B | sam21pp-S | sam21pp-T")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--out", default="eval/results/dam4sam")
    args = ap.parse_args()

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
    results = {}
    for name in args.configs:
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
        print(name, results[name], flush=True)
    # HOTA / MOTA / IDF1 của nhãn hiện có trong workspace (sau lần lan truyền cuối) — tham khảo
    results["trackeval_workspace"] = te.evaluate_workspace(store, "2d", scenes=args.scenes)["overall"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dam4sam.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | Giây |", "|---|---|---|---|---|---|"]
    for name in args.configs:
        v = results[name]
        lines.append(f"| {name} | {v['correct']} | {v['wrong']} | {v['id_switch']} | {v['lost']} | {v['seconds']} |")
    (out / "dam4sam.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
