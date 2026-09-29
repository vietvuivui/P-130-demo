"""Chạy và so sánh các mô hình 3D đã huấn luyện sẵn (MMDetection3D) trên các scene val của nuScenes có trên máy.

Chạy trong môi trường riêng `.venv-mm3d` (Python 3.10, xem tools3d/README.md), từ thư mục gốc của repo:

    .venv-mm3d\\Scripts\\python tools3d/run3d.py all --dataroot ..\\v1.0-trainval

Từng bước:
    check                      kiểm tra GPU, PyTorch, mmcv (có op CUDA), mmdet3d
    add-data --dataroot D --src S   chép file các scene val từ thư mục nuScenes khác (vd. v1.0-mini) vào D
    infos  --dataroot D        tạo file info cho các scene val đủ dữ liệu (không cần tạo info cho cả bộ trainval)
    run    --dataroot D -m ... suy luận từng mô hình -> tools3d/work/preds/<mô hình>/.../results_nusc.json
    eval   --dataroot D        chấm mAP/NDS chuẩn nuScenes trên đúng các scene đã chạy, ghi eval/results/det3d/

Chỉ các scene thuộc tập val: mọi trọng số ở đây đều học trên tập train, chấm trên train sẽ bị ảo.
Kết quả nhỏ để đưa vào UI (dự đoán của 3 scene demo) nằm ở eval/results/det3d/<mô hình>/ui_preds.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tools3d" / "work"
OUT = ROOT / "eval" / "results" / "det3d"
UI_SCENES = ["scene-0035", "scene-0097", "scene-0101"]  # 3 scene val chọn ngẫu nhiên (seed 20260928) cho UI 3D
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
MMLAB = "https://download.openmmlab.com/mmdetection3d"

# config: đường dẫn trong mmdet3d/.mim/configs (hoặc trong repo mmdetection3d với repo=True)
MODELS = {
    "pointpillars": dict(
        label="PointPillars (FPN)",
        sensor="LiDAR",
        config="pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py",
        url=f"{MMLAB}/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/"
        "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth",
        paper=dict(mAP=39.7, NDS=53.2),
    ),
    "ssn": dict(
        label="SSN (SECFPN)",
        sensor="LiDAR",
        config="ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_nus-3d.py",
        url=f"{MMLAB}/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/"
        "hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth",
        paper=dict(mAP=40.9, NDS=54.4),
    ),
    "ssn_regnet": dict(
        label="SSN (RegNet-400MF)",
        sensor="LiDAR",
        config="ssn/ssn_hv_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d.py",
        url=f"{MMLAB}/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/"
        "hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth",
        paper=dict(mAP=46.7, NDS=58.2),
    ),
    "centerpoint_pillar": dict(
        label="CenterPoint (pillar 0.2)",
        sensor="LiDAR",
        config="centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
        url=f"{MMLAB}/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
        "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth",
        paper=dict(mAP=48.7, NDS=59.6),
    ),
    "centerpoint_voxel": dict(
        label="CenterPoint (voxel 0.075, backbone SECOND)",
        sensor="LiDAR",
        config="centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
        url=f"{MMLAB}/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
        "centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth",
        paper=dict(mAP=56.5, NDS=65.2),
    ),
    "fcos3d": dict(
        label="FCOS3D (camera)",
        sensor="Camera",
        config="fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py",
        url=f"{MMLAB}/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/"
        "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth",
        paper=dict(mAP=32.1, NDS=39.3),
    ),
    "pgd": dict(
        label="PGD (camera)",
        sensor="Camera",
        config="pgd/pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune.py",
        url=f"{MMLAB}/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune/"
        "pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth",
        paper=dict(mAP=35.8, NDS=42.5),
    ),
    "bevfusion": dict(
        label="BEVFusion (LiDAR + camera)",
        sensor="LiDAR + Camera",
        config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        url=f"{MMLAB}/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth",
        paper=dict(mAP=68.6, NDS=71.4),
        repo=True,
    ),
}
DEFAULT_MODELS = ["pointpillars", "ssn", "centerpoint_pillar", "centerpoint_voxel", "fcos3d", "pgd"]


def mim_dir() -> Path:
    import mmdet3d

    return Path(mmdet3d.__file__).parent / ".mim"


def load_mim_module(rel: str):
    """Module trong mmdet3d/.mim/tools (không phải package cài sẵn nên nạp theo đường dẫn)."""
    path = mim_dir() / "tools" / rel
    spec = importlib.util.spec_from_file_location(Path(rel).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    print(f"  tải {dest.name} ...", flush=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------------------------
def cmd_check(args) -> None:
    import torch

    print("python ", sys.version.split()[0])
    print("torch  ", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU    ", p.name, f"{p.total_memory / 2**30:.1f} GB")
    import mmcv
    import mmdet
    import mmdet3d
    import mmengine
    from mmcv.ops import Voxelization  # noqa: F401  op CUDA mà mô hình LiDAR cần

    print(
        "mmengine",
        mmengine.__version__,
        "| mmcv",
        mmcv.__version__,
        "| mmdet",
        mmdet.__version__,
        "| mmdet3d",
        mmdet3d.__version__,
    )
    import nuscenes  # noqa: F401

    print("nuscenes-devkit OK | configs:", mim_dir() / "configs")
    if not torch.cuda.is_available():
        raise SystemExit("Không thấy GPU: cài lại torch bản cu118 (xem tools3d/README.md)")


def load_nusc(dataroot: Path, version: str):
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(dataroot), verbose=True)


def complete_val_scenes(nusc, dataroot: Path, max_sweeps: int = 10) -> list[str]:
    """Scene thuộc tập val có đủ file: LiDAR keyframe + 10 lần quét trước, ảnh keyframe 6 camera.

    Bản trainval tải thiếu blob chỉ có một phần scene; đọc danh sách thư mục một lần thay vì kiểm từng file."""
    from nuscenes.utils.splits import create_splits_scenes

    val = set(create_splits_scenes()["val"])
    have = set()
    for d in ["samples/LIDAR_TOP", "sweeps/LIDAR_TOP"] + [f"samples/{c}" for c in CAMERAS]:
        p = dataroot / d
        if p.is_dir():
            have.update(f"{d}/{n}" for n in os.listdir(p))
    out = []
    for scene in nusc.scene:
        if scene["name"] not in val:
            continue
        ok, tok = True, scene["first_sample_token"]
        while tok and ok:
            s = nusc.get("sample", tok)
            sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            files = [sd["filename"]] + [nusc.get("sample_data", s["data"][c])["filename"] for c in CAMERAS]
            prev = sd["prev"]
            for _ in range(max_sweeps):
                if not prev:
                    break
                p = nusc.get("sample_data", prev)
                files.append(p["filename"])
                prev = p["prev"]
            ok = all(f in have for f in files)
            tok = s["next"]
        if ok:
            out.append(scene["name"])
    return sorted(out)


class _SampleSubset:
    """Giống đối tượng NuScenes nhưng `sample` chỉ gồm các keyframe được chọn; mọi thứ khác dùng bảng đầy đủ."""

    def __init__(self, nusc, samples):
        self._nusc = nusc
        self.sample = samples

    def __getattr__(self, name):
        return getattr(self._nusc, name)


def cmd_add_data(args) -> None:
    """Chép file của các scene val từ một thư mục nuScenes khác (vd. v1.0-mini) vào --dataroot.

    Bảng của v1.0-trainval đã có đủ 850 scene (gồm cả các scene của mini), chỉ thiếu file; chép file vào là runner tự
    nhận các scene đó. Chỉ chép LiDAR (keyframe + sweep) và 6 camera keyframe của scene val, bỏ qua file đã có."""
    import shutil

    from nuscenes.utils.splits import create_splits_scenes

    src = args.src.resolve()
    tables = src / args.src_version
    load = lambda name: json.loads((tables / f"{name}.json").read_text(encoding="utf-8"))  # noqa: E731
    val = set(create_splits_scenes()["val"])
    scenes = {s["token"]: s["name"] for s in load("scene") if s["name"] in val}
    samples = {s["token"] for s in load("sample") if s["scene_token"] in scenes}
    channel = {c["token"]: c["sensor_token"] for c in load("calibrated_sensor")}
    sensor = {s["token"]: s["channel"] for s in load("sensor")}
    files = []
    for sd in load("sample_data"):
        if sd["sample_token"] not in samples:
            continue
        ch = sensor[channel[sd["calibrated_sensor_token"]]]
        if ch == "LIDAR_TOP" or (ch in CAMERAS and sd["is_key_frame"]):
            files.append(sd["filename"])
    print(f"{len(scenes)} scene val trong {src.name}: {', '.join(sorted(scenes.values()))}; {len(files)} file")
    copied = 0
    for i, f in enumerate(files, 1):
        a, b = src / f, args.dataroot / f
        if not a.is_file():
            continue
        if not b.is_file() or b.stat().st_size != a.stat().st_size:
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a, b)
            copied += 1
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print(f"Đã chép {copied} file mới vào {args.dataroot} (bỏ qua {len(files) - copied} file đã có hoặc thiếu ở nguồn)")


def cmd_infos(args) -> None:
    import mmengine

    nusc = load_nusc(args.dataroot, args.version)
    ok = complete_val_scenes(nusc, args.dataroot)
    print(f"{len(ok)} scene val đủ dữ liệu: {', '.join(ok)}")
    if args.scenes:
        scenes = args.scenes
    else:
        ui = [s for s in UI_SCENES if s in ok]
        rest = [s for s in ok if s not in ui]
        n = max(0, (args.max_scenes or len(ok)) - len(ui))
        scenes = sorted(ui + random.Random(args.seed).sample(rest, min(n, len(rest))))
    missing = sorted(set(scenes) - set(ok))
    if missing:
        raise SystemExit(f"Scene thiếu file hoặc không thuộc val: {', '.join(missing)}")
    tokens = {s["token"] for s in nusc.scene if s["name"] in scenes}
    # _fill_trainval_infos duyệt nusc.sample: đưa cho nó một lớp bọc chỉ có keyframe của scene đã chọn. Không được gán
    # lại nusc.sample, vì nusc.get() tra theo chỉ số trong bảng đầy đủ và sẽ trả nhầm sample (lỗi "assert 1, 37").
    conv = load_mim_module("dataset_converters/nuscenes_converter.py")
    upd = load_mim_module("dataset_converters/update_infos_to_v2.py")
    subset = _SampleSubset(nusc, [s for s in nusc.sample if s["scene_token"] in tokens])
    _, infos = conv._fill_trainval_infos(subset, set(), tokens, False, max_sweeps=10)
    # mmdet3d tách tên file sweep theo os.sep: trên Windows phải chuẩn hoá "\\"
    for info in infos:
        info["lidar_path"] = osp.normpath(info["lidar_path"])
        for sw in info["sweeps"]:
            sw["data_path"] = osp.normpath(sw["data_path"])
        for cam in info["cams"].values():
            cam["data_path"] = osp.normpath(cam["data_path"])
    v1 = WORK / "infos" / "v1" / "nuscenes_infos_val.pkl"
    v1.parent.mkdir(parents=True, exist_ok=True)
    mmengine.dump(dict(infos=infos, metadata=dict(version=args.version)), str(v1))
    upd.NuScenes = lambda *a, **k: nusc  # bản gốc nạp lại bảng từ ./data/nuscenes cố định
    upd.update_nuscenes_infos(str(v1), str(WORK / "infos"))
    meta = dict(dataroot=str(args.dataroot), version=args.version, scenes=scenes, n_samples=len(infos))
    (WORK / "infos" / "scenes.json").write_text(json.dumps(meta, indent=1))
    print(f"Đã tạo info: {len(scenes)} scene, {len(infos)} keyframe -> {WORK / 'infos'}")


# ---------------------------------------------------------------------------------------------
def _set_eval(ev, ann: str, data_root: str, prefix: str):
    evs = ev if isinstance(ev, list) else [ev]
    for e in evs:
        e.update(data_root=data_root, ann_file=ann, format_only=True, jsonfile_prefix=prefix)
    return ev


def run_model(name: str, args) -> dict:
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    m = MODELS[name]
    if m.get("repo"):
        if not args.mmdet3d_repo:
            raise SystemExit(f"{name} cần repo mmdetection3d đã build op (--mmdet3d-repo), xem tools3d/README.md")
        repo = Path(args.mmdet3d_repo).resolve()
        sys.path.insert(0, str(repo))
        cfg_path = repo / m["config"]
    else:
        cfg_path = mim_dir() / "configs" / m["config"]
    ckpt = download(m["url"], WORK / "ckpts")
    cfg = Config.fromfile(str(cfg_path))
    data_root = str(args.dataroot) + os.sep
    ann = str(WORK / "infos" / "nuscenes_infos_val.pkl")
    prefix = str(WORK / "preds" / name)
    dl = cfg.test_dataloader
    dl.num_workers = args.workers
    dl.persistent_workers = args.workers > 0
    dl.dataset.data_root = data_root
    dl.dataset.ann_file = ann
    cfg.test_evaluator = _set_eval(cfg.test_evaluator, ann, data_root, prefix)
    cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
    cfg.load_from = str(ckpt) if ckpt else None  # None: trọng số ngẫu nhiên, chỉ dùng khi thử quy trình
    cfg.work_dir = str(WORK / "runs" / name)
    cfg.launcher = "none"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    Runner.from_cfg(cfg).test()
    dt = time.time() - t0
    files = sorted(Path(prefix).rglob("results_nusc.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"{name}: không thấy results_nusc.json trong {prefix}")
    n = json.loads((WORK / "infos" / "scenes.json").read_text())["n_samples"]
    info = dict(
        model=name,
        label=m["label"],
        sensor=m["sensor"],
        paper=m["paper"],
        results=str(files[-1]),
        seconds=round(dt, 1),
        n_samples=n,
        s_per_sample=round(dt / max(n, 1), 3),
        peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
    )
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "run.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(f"{name}: {dt:.0f}s ({info['s_per_sample']} s/keyframe, GPU {info['peak_gpu_gb']} GB)", flush=True)
    return info


def cmd_run(args) -> None:
    if not (WORK / "infos" / "nuscenes_infos_val.pkl").is_file():
        raise SystemExit("Chưa có info: chạy `run3d.py infos --dataroot ...` trước")
    failed = {}
    for name in args.models:
        try:
            run_model(name, args)
        except Exception as e:  # một mô hình lỗi (thiếu op, hết VRAM...) không chặn các mô hình khác
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"!! {name} lỗi: {failed[name]}", flush=True)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            (OUT / name / "run.json").write_text(json.dumps(dict(model=name, error=failed[name]), ensure_ascii=False))
    if failed:
        print("Mô hình lỗi:", json.dumps(failed, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------
def subset_eval(nusc, result_path: str, out_dir: str):
    """DetectionEval chuẩn nuScenes nhưng chỉ trên các keyframe có trong file kết quả (tập scene val con).

    Bản gốc đòi file kết quả phủ toàn bộ tập val (150 scene)."""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import DetectionEval

    ev = DetectionEval.__new__(DetectionEval)
    ev.nusc, ev.result_path, ev.eval_set, ev.output_dir, ev.verbose = nusc, result_path, "val", out_dir, False
    ev.cfg = config_factory("detection_cvpr_2019")
    ev.plot_dir = osp.join(out_dir, "plots")
    os.makedirs(ev.plot_dir, exist_ok=True)
    pred, ev.meta = load_prediction(result_path, ev.cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    gt_all = load_gt(nusc, "val", DetectionBox, verbose=False)
    gt = EvalBoxes()
    for t in pred.sample_tokens:
        gt.add_boxes(t, gt_all[t])
    ev.pred_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, pred), ev.cfg.class_range, verbose=False)
    ev.gt_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, gt), ev.cfg.class_range, verbose=False)
    ev.sample_tokens = ev.gt_boxes.sample_tokens
    metrics, _ = ev.evaluate()
    return ev, metrics.serialize()


def pr_by_threshold(ev, thresholds=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), dist=2.0) -> dict:
    """Precision / recall ở từng ngưỡng điểm (khớp cùng lớp, tâm cách < 2 m), để chọn ngưỡng dùng trong sản phẩm."""
    import numpy as np

    tp, fp, n_gt = np.zeros(len(thresholds)), np.zeros(len(thresholds)), 0
    for tok in ev.gt_boxes.sample_tokens:
        preds = sorted(ev.pred_boxes[tok], key=lambda b: -b.detection_score)
        gts, used = ev.gt_boxes[tok], set()
        n_gt += len(gts)
        for p in preds:
            best, bd = None, dist
            for i, g in enumerate(gts):
                if i in used or g.detection_name != p.detection_name:
                    continue
                d = float(np.hypot(p.translation[0] - g.translation[0], p.translation[1] - g.translation[1]))
                if d < bd:
                    best, bd = i, d
            hit = best is not None
            if hit:
                used.add(best)
            for k, t in enumerate(thresholds):
                if p.detection_score >= t:
                    tp[k] += hit
                    fp[k] += not hit
    return dict(
        thresholds=list(thresholds),
        precision=(tp / np.maximum(tp + fp, 1)).round(4).tolist(),
        recall=(tp / max(n_gt, 1)).round(4).tolist(),
        n_gt=n_gt,
    )


def compact_box(b: dict) -> dict:
    """Làm tròn box nuScenes (mm, 1e-5 cho quaternion) để file dự đoán đưa vào git nhỏ lại một nửa."""
    r = lambda v, n: [round(x, n) for x in v]  # noqa: E731
    return dict(
        b, translation=r(b["translation"], 3), size=r(b["size"], 3), rotation=r(b["rotation"], 5),
        velocity=r(b["velocity"], 3), detection_score=round(b["detection_score"], 4),
    )  # fmt: skip


def cmd_eval(args) -> None:
    nusc = load_nusc(args.dataroot, args.version)
    ui_tokens = {}
    for s in nusc.scene:
        if s["name"] in UI_SCENES:
            tok = s["first_sample_token"]
            while tok:
                ui_tokens[tok] = s["name"]
                tok = nusc.get("sample", tok)["next"]
    summary = {}
    for name in args.models:
        run_file = OUT / name / "run.json"
        if not run_file.is_file():
            continue
        run = json.loads(run_file.read_text(encoding="utf-8"))
        if "error" in run:
            summary[name] = run
            continue
        ev, metrics = subset_eval(nusc, run["results"], str(WORK / "eval" / name))
        pr = pr_by_threshold(ev)
        (OUT / name / "metrics.json").write_text(json.dumps(dict(metrics=metrics, pr=pr), indent=1))
        # dự đoán của 3 scene demo, bỏ box điểm rất thấp: đủ nhỏ để đưa vào repo và UI
        res = json.loads(Path(run["results"]).read_text())["results"]
        ui = {t: [compact_box(b) for b in res[t] if b["detection_score"] >= 0.05] for t in res if t in ui_tokens}
        (OUT / name / "ui_preds.json").write_text(json.dumps(dict(scenes=UI_SCENES, results=ui), separators=(",", ":")))
        summary[name] = dict(
            run,
            mAP=metrics["mean_ap"],
            NDS=metrics["nd_score"],
            ap=metrics["mean_dist_aps"],
            tp_errors=metrics["tp_errors"],
            pr=pr,
        )
        print(f"{name:20s} mAP {metrics['mean_ap']:.3f}  NDS {metrics['nd_score']:.3f}", flush=True)
    meta = json.loads((WORK / "infos" / "scenes.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(
            dict(scenes=meta["scenes"], n_samples=meta["n_samples"], models=summary), indent=1, ensure_ascii=False
        )
    )
    print(f"Đã ghi {OUT / 'summary.json'}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # PowerShell ghi ra file bằng cp1252: in tiếng Việt sẽ lỗi
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["check", "add-data", "infos", "run", "eval", "all"])
    ap.add_argument("--dataroot", type=Path, help="Thư mục nuScenes (chứa v1.0-trainval/, samples/, sweeps/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--src", type=Path, help="add-data: thư mục nuScenes nguồn (vd. ..\\v1.0-mini)")
    ap.add_argument("--src-version", default="v1.0-mini", help="add-data: tên thư mục bảng của nguồn")
    ap.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    ap.add_argument("--scenes", nargs="+", help="Chỉ định scene val; mặc định mọi scene val đủ dữ liệu")
    ap.add_argument("--max-scenes", type=int, help="Giới hạn số scene (luôn gồm 3 scene demo)")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--workers", type=int, default=0, help="num_workers của dataloader (Windows: để 0)")
    ap.add_argument("--mmdet3d-repo", help="Repo mmdetection3d đã build op, chỉ cần cho bevfusion")
    args = ap.parse_args()
    if args.cmd != "check" and not args.dataroot:
        ap.error("cần --dataroot")
    if args.cmd == "add-data" and not args.src:
        ap.error("add-data cần --src")
    if args.dataroot:
        args.dataroot = args.dataroot.resolve()
    steps = ["check", "infos", "run", "eval"] if args.cmd == "all" else [args.cmd]
    for step in steps:
        print(f"\n===== {step} =====", flush=True)
        globals()[f"cmd_{step.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
