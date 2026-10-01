"""Chấm detector 2D (CAM_FRONT) trên keyframe nuScenes val, tách dev (3 scene demo của UI) và held-out (mọi scene val
khác có ảnh trên máy). Dùng để so trọng số gốc với trọng số đã fine-tune một cách công bằng.

    python tools2d/eval2d.py --dataroot ..\\v1.0-trainval --weights yoloe-26l-seg.pt weights\\yoloe-26l-nuimages.pt

Mỗi --weights là một cấu hình YOLOE (giữ nguyên mọi thứ khác của configs/autolabel.yaml). In mAP50 theo từng lớp,
precision / recall ở ngưỡng detection.min_score, ghi eval/results/det2d_finetune.json.

Lưu ý khi đọc số: GT 2D của nuScenes là hộp bao của box 3D chiếu xuống ảnh (rộng hơn hộp sát vật), còn nuImages gán hộp
sát vật. Mô hình học trên nuImages vẽ box chặt hơn nên IoU với GT chiếu có thể giảm dù phát hiện đúng hơn: xem cả recall
và AP từng lớp, và kết quả trên nuImages val (Ultralytics in ra khi huấn luyện).
Chỉ quyết định dùng trọng số mới khi held-out tốt hơn — dev đã được nhìn nhiều lần khi làm UI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEV = ["scene-0035", "scene-0097", "scene-0101"]


def evaluate(dets_per_img, gts, min_score: float) -> dict:
    from src.services.evaluation import _match, average_precision

    per, n_gt = defaultdict(list), defaultdict(int)
    tp = fp = n_all = 0
    for dets, gt in zip(dets_per_img, gts, strict=True):
        for g in gt:
            if not g["ignore"]:
                n_gt[g["label"]] += 1
                n_all += 1
        preds = sorted(({"bbox": d.bbox, "label": d.label, "score": d.score} for d in dets if not d.label.startswith("__")),
                       key=lambda p: -p["score"])  # fmt: skip
        for lb in {p["label"] for p in preds}:
            cp = [p for p in preds if p["label"] == lb]
            st, _ = _match(cp, gt, 0.5)
            per[lb] += [(p["score"], s == "tp") for p, s in zip(cp, st, strict=True) if s != "ignore"]
            tp += sum(1 for p, s in zip(cp, st, strict=True) if s == "tp" and p["score"] >= min_score)
            fp += sum(1 for p, s in zip(cp, st, strict=True) if s == "fp" and p["score"] >= min_score)
    ap = {}
    for lb in n_gt:
        items = sorted(per.get(lb, []), key=lambda x: -x[0])
        ap[lb] = average_precision(np.array([t for _, t in items], float), n_gt[lb])
    p, r = tp / max(tp + fp, 1), tp / max(n_all, 1)
    return dict(mAP50=round(float(np.mean(list(ap.values()))), 4), P=round(p, 4), R=round(r, 4),
                F1=round(2 * p * r / max(p + r, 1e-9), 4), ap={k: round(v, 3) for k, v in sorted(ap.items())},
                n_gt=dict(n_gt), images=len(gts))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "autolabel.yaml")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache_eval2d")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "det2d_finetune.json")
    args = ap.parse_args()

    from nuscenes_val import VAL_SCENES

    from src.models.qa_config import load_autolabel_config
    from src.services.detectors import DetectorEnsemble
    from src.services.nuscenes_data import NuScenesMini

    base = load_autolabel_config(args.config)
    print("Đọc bảng nuScenes (bộ trainval đầy đủ mất vài chục giây)…", flush=True)
    data = NuScenesMini(args.dataroot, args.version)
    splits = defaultdict(list)
    for scene, idx, tok in data.keyframes(sorted(VAL_SCENES)):
        f = data.camera_frame(tok, "CAM_FRONT", [], scene, idx)
        path = args.dataroot / f.image.path
        if path.exists():
            gt = data.gt_boxes_2d(f, base.gt_category_map)
            splits["dev" if scene in DEV else "heldout"].append((f.image.sd_token, path, gt))
    if not splits:
        raise SystemExit("Không có ảnh CAM_FRONT nào của scene val trên máy")
    print({k: len(v) for k, v in splits.items()}, "keyframe")

    res = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for w in args.weights:
        cfg = base.model_copy(deep=True)
        cfg.detection.detectors = ["yoloe"]
        cfg.detection.yoloe.weights = w
        if args.imgsz:
            cfg.detection.yoloe.imgsz = args.imgsz
        ens = DetectorEnsemble(cfg, args.cache)
        for split, items in sorted(splits.items()):
            t0 = time.time()
            dets = []
            for i in range(0, len(items), 8):
                dets += ens.detect_batch([(tok, p) for tok, p, _ in items[i : i + 8]])
            r = evaluate(dets, [g for _, _, g in items], cfg.detection.min_score)
            r["sec"] = round(time.time() - t0, 1)
            res[f"{Path(w).name}@{cfg.detection.yoloe.imgsz}:{split}"] = r
            print(f"{Path(w).name:32s} {split:8s} mAP50 {r['mAP50']:.3f}  P {r['P']:.3f}  R {r['R']:.3f}  F1 {r['F1']:.3f}  "
                  f"({r['images']} ảnh)", flush=True)  # fmt: skip
            print("   AP:", "  ".join(f"{k} {v:.2f}" for k, v in r["ap"].items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Ghi", args.out)


if __name__ == "__main__":
    main()
