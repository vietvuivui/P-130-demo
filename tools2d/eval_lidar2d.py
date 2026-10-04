"""Chấm bước gộp box 3D (ensemble LiDAR) vào nhãn 2D (src/services/lidar2d.py) với các cách hạ điểm box chỉ camera thấy.

    python tools2d/eval_lidar2d.py --dataroot ..\\v1.0-trainval

Dùng dự đoán 3D đã có của 4 mô hình LiDAR (tools3d/work/preds/<model>/pred_instances_3d/results_nusc.json, do
scripts\\tasks.ps1 eval3d sinh) -> gộp + tinh chỉnh theo track (tools3d/refine3d.py, CPU) -> chiếu xuống ảnh camera; detection
2D lấy từ cache của tools2d/eval2d.py (thiếu thì chạy model, GPU). So với nhãn gốc nuScenes, tách dev (3 scene demo) và
test (các scene val còn lại có dự đoán 3D). In mAP50 / P / R / F1 và số box phải sửa cho từng cấu hình:

    yoloe                       : chỉ detector ảnh
    fuse, box 0 điểm như thường : gộp, mọi box chỉ camera thấy x camera_only_scale (mặc định trước 04/10/2026)
    fuse (mặc định)             : như trên, riêng box chỉ camera thấy mà không có điểm LiDAR trong box x no_lidar_scale

--cameras all chấm cả 6 camera (in thêm từng camera của tập test); --all-variants chạy thêm các phép dò cũ.
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
sys.path.insert(0, str(ROOT / "tools3d"))

LIDAR_MODELS = ["centerpoint_voxel", "centerpoint_pillar", "ssn", "pointpillars"]
DEV = ["scene-0035", "scene-0097", "scene-0101"]
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# no_lidar_scale: "cfg" = theo detection.lidar3d.no_lidar_scale của config; None = box 0 điểm LiDAR hạ như box chỉ camera khác
VARIANTS = {
    "yoloe": None,
    "fuse, box 0 điểm như thường": dict(no_lidar_scale=None),  # mặc định trước 04/10/2026
    "fuse (mặc định)": dict(no_lidar_scale="cfg"),
}
# Các phép dò cũ (khử box 3D trùng, lớp theo camera), chạy thêm bằng --all-variants
EXTRA_VARIANTS = {
    "fuse, không khử trùng": dict(no_lidar_scale=None, dedup=None),
    # Dò ngưỡng chồng nhau trên mặt đường (xe tải ở scene-0035_023 chồng 22%); "khác lớp" = chỉ bỏ cặp box khác lớp
    "fuse khử trùng 0.20": dict(no_lidar_scale=None, dedup=0.20),
    "fuse khử trùng 0.30": dict(no_lidar_scale=None, dedup=0.30),
    "fuse khử trùng 0.50": dict(no_lidar_scale=None, dedup=0.50),
    "fuse khử khác lớp 0.50": dict(no_lidar_scale=None, dedup=0.50, cross_only=True),
    # Cặp box khác lớp ở cùng chỗ: lấy lớp của detector ảnh khi nó có box ở đó (thay vì lớp có điểm cao hơn)
    "fuse 0.15, lớp theo camera": dict(no_lidar_scale=None, dedup=0.15, camera_label=True),
}


class _Nusc:
    """Đủ cho refine3d.scenes_from_nusc: get("sample", token)."""

    def __init__(self, data):
        self.data = data

    def get(self, table, tok):
        assert table == "sample"
        return self.data.sample[tok]


def load_ensemble(work: Path, data) -> dict[str, list[dict]]:
    import refine3d

    results = {}
    for m in LIDAR_MODELS:
        f = work / "preds" / m / "pred_instances_3d" / "results_nusc.json"
        if f.is_file():
            results[m] = json.loads(f.read_text())["results"]
            print(f"  3D {m}: {len(results[m])} sample", flush=True)
    if not results:
        raise SystemExit(f"Không có dự đoán 3D trong {work / 'preds'} (chạy scripts\\tasks.ps1 eval3d trước)")
    fused = refine3d.fuse(results, min_score=0.01)
    scenes = refine3d.scenes_from_nusc(_Nusc(data), list(fused))
    return refine3d.refine_tracks(fused, scenes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "autolabel.yaml")
    ap.add_argument("--work", type=Path, default=ROOT / "tools3d" / "work")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache_eval2d")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "improve" / "lidar2d.json")
    ap.add_argument("--scenes", nargs="*", help="chỉ các scene này (mặc định: mọi scene val có dự đoán 3D)")
    ap.add_argument("--cameras", nargs="*", default=["CAM_FRONT"], help="camera cần chấm; 'all' = cả 6 camera")
    ap.add_argument("--all-variants", action="store_true", help="chạy thêm các phép dò cũ (khử trùng, lớp theo camera)")
    args = ap.parse_args()
    cameras = CAMERAS if args.cameras == ["all"] else args.cameras
    variants = {**VARIANTS, **(EXTRA_VARIANTS if args.all_variants else {})}

    from eval2d import evaluate
    from nuscenes_val import VAL_SCENES

    from src.models.qa_config import load_autolabel_config
    from src.services.detectors import DetectorEnsemble
    from src.services.lidar2d import dedup_bev, merge_detections, project_boxes
    from src.services.nuscenes_data import NuScenesMini

    cfg = load_autolabel_config(args.config)
    cfg.detection.detectors = ["yoloe"]
    print("Đọc bảng nuScenes…", flush=True)
    data = NuScenesMini(args.dataroot, args.version)
    preds = load_ensemble(args.work, data)
    ens = DetectorEnsemble(cfg, args.cache)
    l3 = cfg.detection.lidar3d

    frames = []
    for scene, idx, tok in data.keyframes(sorted(VAL_SCENES)):
        if tok not in preds or (args.scenes and scene not in args.scenes):
            continue
        for cam in cameras:
            f = data.camera_frame(tok, cam, [], scene, idx)
            path = args.dataroot / f.image.path
            if path.exists():
                frames.append((scene, tok, cam, f, path))
    print(f"{len(frames)} ảnh ({len(cameras)} camera); detect (đọc cache, thiếu thì chạy model)…", flush=True)
    t0 = time.time()
    all_dets = []
    for i in range(0, len(frames), 8):
        all_dets += ens.detect_batch([(f.image.sd_token, path) for _, _, _, f, path in frames[i : i + 8]])
    print(f"detect xong: {time.time() - t0:.0f}s, {getattr(ens, 'model_images', 0)} ảnh phải chạy model", flush=True)

    items: list[tuple] = []  # (split, cam, dets, (box 3D thô, box 3D đã khử trùng), uv, gt)
    for (scene, tok, cam, f, _), dets in zip(frames, all_dets, strict=True):
        gt = data.gt_boxes_2d(f, cfg.gt_category_map)
        boxes3d = project_boxes(
            preds[tok], data.cam_from_global(f.image.sd_token), f.intrinsic, f.image.width, f.image.height, l3.min_score,
            ego_from_global=np.linalg.inv(data._global_from_ego(f.image.sd_token)),
        )  # fmt: skip
        boxes3d = (boxes3d, dedup_bev(boxes3d, l3.dedup_bev_overlap) if l3.dedup_bev_overlap is not None else boxes3d)
        uv, _ = data.lidar_in_image(f, cfg.qa.lidar.min_depth_m) if f.lidar_sd_token else (None, None)
        items.append(("dev" if scene in DEV else "test", cam, dets, boxes3d, uv, gt))
    count = defaultdict(int)
    for split, *_ in items:
        count[split] += 1
    print(dict(count), f"ảnh keyframe ({time.time() - t0:.0f}s)", flush=True)

    res = {"_run": {"work": str(args.work), "models": LIDAR_MODELS, "cameras": cameras,
                    "weights": cfg.detection.yoloe.weights, "camera_only_scale": l3.camera_only_scale,
                    "match_iou": l3.match_iou, "no_lidar_scale": l3.no_lidar_scale,
                    "no_lidar_max_points": l3.no_lidar_max_points}}  # fmt: skip
    print(f"{'cấu hình':<28} {'tập':<22} {'mAP50':>6} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'phải sửa':>9}")
    for name, opt in variants.items():
        merged = []
        for _, _, dets, (raw3d, dedup3d), uv, _ in items:
            if opt is None:
                merged.append(dets)
                continue
            if opt.get("dedup") is not None:
                boxes3d = dedup_bev([dict(b) for b in raw3d], opt["dedup"], cross_only=opt.get("cross_only", False))
            else:
                boxes3d = [dict(b) for b in (raw3d if "dedup" in opt else dedup3d)]
            nls = l3.no_lidar_scale if opt["no_lidar_scale"] == "cfg" else opt["no_lidar_scale"]
            merged.append(merge_detections(dets, boxes3d, l3.match_iou, l3.camera_only_scale, uv=uv, no_lidar_scale=nls,
                                           camera_label_wins=opt.get("camera_label", False),
                                           no_lidar_max_points=l3.no_lidar_max_points))  # fmt: skip
        groups = [(split, None) for split in sorted(count)]
        if len(cameras) > 1:  # thêm từng camera của tập test
            groups += [("test", cam) for cam in cameras]
        for split, cam in groups:
            keep = [k for k, it in enumerate(items) if it[0] == split and (cam is None or it[1] == cam)]
            if not keep:
                continue
            r = evaluate([merged[k] for k in keep], [items[k][5] for k in keep], cfg.detection.min_score)
            n_gt = sum(r["n_gt"].values())
            tp = round(r["R"] * n_gt)
            fp = round(tp / max(r["P"], 1e-9) - tp) if r["P"] else 0
            fn = n_gt - tp
            r.update(TP=tp, FP=fp, FN=fn, fix=fp + fn)
            tag = split if cam is None else f"{split}/{cam}"
            res[f"{name}:{tag}"] = r
            print(f"{name:<28} {tag:<22} {r['mAP50']:>6.3f} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f} "
                  f"{tp:>5} {fp:>5} {fn:>5} {fp + fn:>9}", flush=True)  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Ghi", args.out)
    print("phải sửa = box thừa phải xoá (FP) + vật sót phải vẽ (FN) ở ngưỡng giữ", cfg.detection.min_score)


if __name__ == "__main__":
    main()
