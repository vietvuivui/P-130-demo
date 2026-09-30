"""Đo lan truyền box 3D đã duyệt (thí nghiệm keyframe hoàn hảo, src/services/propagation3d_eval.py) trên các scene val
có dự đoán của `run3d.py run`: dev = 3 scene demo, held-out = các scene còn lại. So ba cách:

    ego_only        chỉ bù chuyển động xe (box đứng yên trong hệ toàn cục)
    velocity        + dịch theo vận tốc mô hình dự đoán, ngưỡng khớp theo lớp của tracker CenterPoint
    velocity_tuned  + ngưỡng ×0.5, chịu mất 2 keyframe (chọn trên dev; mặc định trong configs/autolabel.yaml)

    python tools3d/eval_propagation3d.py --dataroot ..\\v1.0-trainval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEV = ["scene-0035", "scene-0097", "scene-0101"]
LIDAR = ["pointpillars", "ssn", "centerpoint_pillar", "centerpoint_voxel"]
VARIANTS = {
    "ego_only": dict(motion="none", max_misses=1, dist_scale=1.0),
    "velocity": dict(motion="velocity", max_misses=1, dist_scale=1.0),
    "velocity_tuned": dict(motion="velocity", max_misses=2, dist_scale=0.5),
}


def load_preds(work: Path, model: str) -> dict:
    files = sorted((work / "preds" / model).rglob("results_nusc.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"Không thấy dự đoán của {model} trong {work / 'preds'} (chạy tools3d/run3d.py run trước)")
    return json.loads(files[-1].read_text(encoding="utf-8"))["results"]


def main() -> None:
    import refine3d

    from src.models.qa_config import Propagation3DCfg, load_autolabel_config
    from src.services.nuscenes_data import NuScenesMini
    from src.services.propagation3d_eval import evaluate, scene_keyframes

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--work", type=Path, default=ROOT / "tools3d" / "work")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "propagation3d.json")
    args = ap.parse_args()

    config = load_autolabel_config(ROOT / "configs" / "autolabel.yaml")
    data = NuScenesMini(args.dataroot, args.version)
    preds = {m: load_preds(args.work, m) for m in LIDAR}
    have = set(preds["centerpoint_voxel"])
    scenes = [s["name"] for s in data.scene.values() if s["first_sample_token"] in have]
    splits = {"dev": [s for s in scenes if s in DEV], "heldout": sorted(s for s in scenes if s not in DEV)}
    sc = {s: [(t, ts / 1e6) for t, ts in scene_keyframes(data, s)] for s in scenes}
    ens = refine3d.refine_tracks(refine3d.fuse(preds, min_score=0.01), sc)
    out = {}
    for split, names in splits.items():
        out[split] = {"scenes": names}
        print(f"\n== {split}: {len(names)} scene ==")
        print(f"{'mô hình / cách':<34} {'đúng':>6} {'box ra':>7} {'tỉ lệ':>6} {'đổi ID':>7} {'sai':>5} {'mất':>6} {'phủ':>6}")
        for mname, model_preds in (("ensemble", ens), ("centerpoint_voxel", preds["centerpoint_voxel"])):
            for vn, kw in VARIANTS.items():
                r = evaluate(data, model_preds, names, Propagation3DCfg(**kw), config.gt_category_map)
                out[split][f"{mname}/{vn}"] = r
                print(f"{mname + ' / ' + vn:<34} {r['correct']:>6} {r['outputs']:>7} {r['correct_rate'] or 0:>6.3f} "
                      f"{r['id_switch']:>7} {r['wrong']:>5} {r['lost']:>6} {r['coverage'] or 0:>6.3f}")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
