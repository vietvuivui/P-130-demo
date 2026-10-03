"""Đánh giá tracking (HOTA, MOTA, IDF1) cho nhãn lan truyền 2D và box 3D, theo chuẩn TrackEval (Luiten et al. 2021).

Hai cách dùng:
- `evaluate_tracks(gt, pred, similarity)`: bộ đo viết lại trong repo (HOTA, DetA, AssA, MOTA, IDF1, IDSW), không cần cài
  thêm gì; cùng công thức với TrackEval (đã đối chiếu, tests/test_services/test_trackeval.py).
- `export_mot(...)`: ghi GT và kết quả ra thư mục dạng MOTChallenge để chạy TrackEval chính thức
  (`pip install git+https://github.com/JonathonLuiten/TrackEval.git`, lệnh `scripts\\tasks.ps1 trackeval`).

Khác bộ đo detection (mAP, P/R): ở đây mỗi vật là một chuỗi theo thời gian; bộ đo phạt cả việc đổi ID giữa chừng
(cùng một xe mà hai keyframe mang hai track_id) và việc ghép hai xe vào một track.

Dữ liệu:
- 2D: GT = nhãn gốc nuScenes (instance_token, box 2D), dự đoán = object của frame có track_id (nhãn lan truyền giữ
  track_id của keyframe gốc; object detector tự sinh có track_id riêng theo frame). Giống nhau = IoU, ngưỡng 0.5.
- 3D: GT = gt3d (instance_token, box 3D ở hệ LiDAR), dự đoán = object của frames3d có track_id. Giống nhau = 1 - d/2m
  với d là khoảng cách tâm trên mặt BEV trong hệ global (nuScenes AMOTA dùng cùng cách khớp 2 m).
"""

from __future__ import annotations

import configparser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo("float").eps


@dataclass
class Obs:
    """Một box ở một frame: id của track (str) và hình học tuỳ chế độ."""

    track: str
    label: str
    geom: Sequence[float]  # 2D: [x1, y1, x2, y2]; 3D: [x, y] tâm BEV (m)
    score: float = 1.0
    ignore: bool = False


@dataclass
class Sequence2D:
    frames: list[tuple[str, list[Obs], list[Obs]]] = field(default_factory=list)  # (frame_id, gt, pred)


def iou_sim(gt: list[Obs], pr: list[Obs]) -> np.ndarray:
    from src.services.geometry import iou_matrix

    if not gt or not pr:
        return np.zeros((len(gt), len(pr)))
    return iou_matrix(np.array([g.geom for g in gt], float), np.array([p.geom for p in pr], float))


def center_sim(radius_m: float = 2.0) -> Callable[[list[Obs], list[Obs]], np.ndarray]:
    def sim(gt: list[Obs], pr: list[Obs]) -> np.ndarray:
        if not gt or not pr:
            return np.zeros((len(gt), len(pr)))
        a = np.array([g.geom[:2] for g in gt], float)[:, None, :]
        b = np.array([p.geom[:2] for p in pr], float)[None, :, :]
        d = np.linalg.norm(a - b, axis=2)
        return np.clip(1.0 - d / radius_m, 0.0, 1.0)

    return sim


def _ids(frames, which: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for fr in frames:
        for o in fr[which]:
            out.setdefault(o.track, len(out))
    return out


def evaluate_tracks(frames: list[tuple[str, list[Obs], list[Obs]]], similarity, thr: float = 0.5,
                    same_class: bool = True) -> dict:  # fmt: skip
    """HOTA / CLEAR / Identity như TrackEval trên một chuỗi frame đã sắp theo thời gian.

    frames: [(frame_id, gt, pred)]; similarity(gt, pred) -> ma trận [0, 1]. same_class: khác lớp coi như không giống.
    GT ignore (vật khó thấy / không có điểm LiDAR) bị bỏ cùng với dự đoán khớp nó (như vùng ignore của MOTChallenge).
    """
    from scipy.optimize import linear_sum_assignment

    # 1) Bỏ GT ignore và dự đoán khớp với chúng (IoU/giống >= thr), rồi đánh số id
    cleaned = []
    for fid, gt, pr in frames:
        keep_gt = [g for g in gt if not g.ignore]
        ign = [g for g in gt if g.ignore]
        if ign and pr:
            s = similarity(ign, pr)
            if same_class:
                s = s * np.array([[g.label == p.label for p in pr] for g in ign])
            drop = set(np.where(s.max(axis=0) >= thr)[0].tolist()) if s.size else set()
            pr = [p for j, p in enumerate(pr) if j not in drop]
        cleaned.append((fid, keep_gt, pr))
    gid, pid = _ids(cleaned, 1), _ids(cleaned, 2)
    ng, npred = len(gid), len(pid)
    sims = []
    for _, gt, pr in cleaned:
        s = similarity(gt, pr) if gt and pr else np.zeros((len(gt), len(pr)))
        if same_class and gt and pr:
            s = s * np.array([[g.label == p.label for p in pr] for g in gt])
        sims.append(s)

    out = {"num_gt_dets": sum(len(f[1]) for f in cleaned), "num_pred_dets": sum(len(f[2]) for f in cleaned),
           "num_gt_ids": ng, "num_pred_ids": npred, "frames": len(cleaned)}  # fmt: skip
    if ng == 0 or npred == 0:
        out.update(
            HOTA=0.0, DetA=0.0, AssA=0.0, MOTA=0.0, IDF1=0.0, IDSW=0, FP=out["num_pred_dets"], FN=out["num_gt_dets"]
        )
        return out

    # 2) HOTA
    pot = np.zeros((ng, npred))
    gcount, pcount = np.zeros(ng), np.zeros(npred)
    for (_, gt, pr), s in zip(cleaned, sims, strict=True):
        gi = [gid[g.track] for g in gt]
        pj = [pid[p.track] for p in pr]
        if gt and pr:
            denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            sim_iou = np.zeros_like(s)
            m = denom > 0 + EPS
            sim_iou[m] = s[m] / denom[m]
            pot[np.ix_(gi, pj)] += sim_iou
        gcount[gi] += 1
        pcount[pj] += 1
    gas = pot / np.maximum(1.0, gcount[:, None] + pcount[None, :] - pot)
    tp = np.zeros(len(ALPHAS))
    fn = np.zeros(len(ALPHAS))
    fp = np.zeros(len(ALPHAS))
    matches = np.zeros((len(ALPHAS), ng, npred))
    for (_, gt, pr), s in zip(cleaned, sims, strict=True):
        gi = [gid[g.track] for g in gt]
        pj = [pid[p.track] for p in pr]
        if not gt or not pr:
            fn += len(gt)
            fp += len(pr)
            continue
        score = gas[np.ix_(gi, pj)] * s
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = s[r, c] >= alpha - EPS
            n = int(ok.sum())
            tp[a] += n
            fn[a] += len(gt) - n
            fp[a] += len(pr) - n
            for rr, cc in zip(r[ok], c[ok], strict=True):
                matches[a, gi[rr], pj[cc]] += 1
    deta = tp / np.maximum(1.0, tp + fn + fp)
    assa = np.zeros(len(ALPHAS))
    for a in range(len(ALPHAS)):
        ass = matches[a] / np.maximum(1.0, gcount[:, None] + pcount[None, :] - matches[a])
        assa[a] = (matches[a] * ass).sum() / max(1.0, tp[a])
    hota = np.sqrt(deta * assa)
    out.update(HOTA=float(hota.mean()), DetA=float(deta.mean()), AssA=float(assa.mean()),
               HOTA_alpha={f"{al:.2f}": float(h) for al, h in zip(ALPHAS, hota, strict=True)})  # fmt: skip

    # 3) CLEAR (MOTA, ngưỡng thr; ưu tiên giữ cặp đã khớp ở frame trước như TrackEval)
    prev: dict[int, int] = {}
    ctp = cfp = cfn = idsw = 0
    for (_, gt, pr), s in zip(cleaned, sims, strict=True):
        gi = [gid[g.track] for g in gt]
        pj = [pid[p.track] for p in pr]
        if not gt or not pr:
            cfn += len(gt)
            cfp += len(pr)
            continue
        score = s.copy()
        for a, g in enumerate(gi):
            for b, p in enumerate(pj):
                if prev.get(g) == p:
                    score[a, b] += 1000.0
        score[s < thr - EPS] = 0
        r, c = linear_sum_assignment(-score)
        ok = s[r, c] >= thr - EPS
        n = int(ok.sum())
        ctp += n
        cfn += len(gt) - n
        cfp += len(pr) - n
        new_prev = dict(prev)
        for rr, cc in zip(r[ok], c[ok], strict=True):
            g, p = gi[rr], pj[cc]
            if g in prev and prev[g] != p:
                idsw += 1
            new_prev[g] = p
        prev = new_prev
    num_gt = ctp + cfn
    out.update(MOTA=float(1.0 - (cfn + cfp + idsw) / max(1, num_gt)), IDSW=idsw, FP=cfp, FN=cfn,
               MOTP=None, Recall=float(ctp / max(1, num_gt)), Precision=float(ctp / max(1, ctp + cfp)))  # fmt: skip

    # 4) Identity (IDF1): ghép id toàn cục một-một
    pot_id = np.zeros((ng, npred))
    for (_, gt, pr), s in zip(cleaned, sims, strict=True):
        if not gt or not pr:
            continue
        gi = [gid[g.track] for g in gt]
        pj = [pid[p.track] for p in pr]
        ok = s >= thr - EPS
        pot_id[np.ix_(gi, pj)] += ok
    n = ng + npred
    fn_mat = np.zeros((n, n))
    fp_mat = np.zeros((n, n))
    fn_mat[:ng, :npred] = gcount[:, None] - pot_id
    fp_mat[:ng, :npred] = pcount[None, :] - pot_id
    big = 1e10
    fn_mat[:ng, npred:] = big
    fp_mat[:ng, npred:] = big
    fn_mat[ng:, :npred] = big
    fp_mat[ng:, :npred] = big
    for i in range(ng):
        fn_mat[i, npred + i] = gcount[i]
        fp_mat[i, npred + i] = 0
    for j in range(npred):
        fp_mat[ng + j, j] = pcount[j]
        fn_mat[ng + j, j] = 0
    r, c = linear_sum_assignment(fn_mat + fp_mat)
    idfn = float(fn_mat[r, c].sum())
    idfp = float(fp_mat[r, c].sum())
    idtp = float(gcount.sum() - idfn)
    out.update(IDF1=float(2 * idtp / max(1.0, 2 * idtp + idfp + idfn)), IDTP=idtp, IDFP=idfp, IDFN=idfn)
    return out


# ---------------------------------------------------------------------------
# Lấy chuỗi từ workspace
# ---------------------------------------------------------------------------


def sequence_2d(store, scene: str, min_score: float = 0.0, propagated_only: bool = False) -> list:
    """[(frame_id, gt, pred)] cho một scene ở chế độ 2D. propagated_only: chỉ chấm frame nhận nhãn lan truyền."""
    frames = sorted((f for f in store.list_frames() if f.scene == scene), key=lambda f: f.index)
    out = []
    for f in frames:
        if propagated_only and not f.propagated_from:
            continue
        gt_raw = store.load_aux("gt", f.frame_id) or []
        gt = [Obs(g["instance_token"], g["label"], g["bbox"], ignore=bool(g.get("ignore"))) for g in gt_raw
              if g.get("instance_token")]  # fmt: skip
        pr = []
        for o in f.objects:
            if o.review.status == "deleted" or (o.source != "human" and o.score < min_score):
                continue
            pr.append(Obs(o.track_id or f"{f.frame_id}:{o.object_id}", o.review.final_label or o.label,
                          o.review.final_bbox or o.bbox, o.score))  # fmt: skip
        out.append((f.frame_id, gt, pr))
    return out


def sequence_3d(store, model: str, scene: str, min_score: float = 0.3) -> list:
    """[(frame_id, gt, pred)] cho một scene ở chế độ 3D; toạ độ tâm đổi sang hệ global để so giữa các frame."""
    frames = sorted((f for f in store.list_frames3d(model) if f.scene == scene), key=lambda f: f.index)
    out = []
    for f in frames:
        g2l = np.array(f.global_from_lidar, float)

        def to_global(c):
            p = g2l @ np.array([c[0], c[1], c[2], 1.0])
            return [float(p[0]), float(p[1])]

        gt_raw = store.load_aux("gt3d", f.frame_id) or []
        gt = [Obs(g["instance_token"], g["label"], to_global(g["center"]), ignore=g.get("num_pts", 1) == 0)
              for g in gt_raw if g.get("instance_token")]  # fmt: skip
        pr = []
        for o in f.objects:
            if o.review.status == "deleted" or (o.source != "human" and o.score < min_score):
                continue
            box = o.review.final_box or o.box
            pr.append(Obs(o.track_id or f"{f.frame_id}:{o.object_id}", o.review.final_label or o.label,
                          to_global(box.center), o.score))  # fmt: skip
        out.append((f.frame_id, gt, pr))
    return out


def evaluate_workspace(store, mode: str = "2d", model: str | None = None, scenes: Sequence[str] | None = None,
                       min_score: float | None = None, propagated_only: bool = False) -> dict:  # fmt: skip
    """Chấm mọi scene (gộp thành một chuỗi dài; id không trùng giữa scene vì instance_token / track_id mang tên scene)."""
    if mode == "2d":
        all_scenes = sorted({f.scene for f in store.list_frames()})
        sim, thr = iou_sim, 0.5
        seqs = {
            s: sequence_2d(store, s, min_score or 0.0, propagated_only) for s in all_scenes if not scenes or s in scenes
        }
    else:
        assert model
        all_scenes = sorted({f.scene for f in store.list_frames3d(model)})
        sim, thr = center_sim(2.0), 0.5
        seqs = {s: sequence_3d(store, model, s, 0.3 if min_score is None else min_score) for s in all_scenes
                if not scenes or s in scenes}  # fmt: skip
    per_scene = {s: evaluate_tracks(fr, sim, thr) for s, fr in seqs.items() if fr}
    overall = evaluate_tracks([x for fr in seqs.values() for x in fr], sim, thr)
    return {"mode": mode, "model": model, "scenes": per_scene, "overall": overall}


# ---------------------------------------------------------------------------
# Xuất MOTChallenge cho TrackEval chính thức
# ---------------------------------------------------------------------------


def export_mot(seqs: dict[str, list], out: Path, tracker_name: str = "autolabel", mode: str = "2d",
               bev_scale: float = 10.0) -> Path:  # fmt: skip
    """Ghi gt/<seq>/gt/gt.txt + seqinfo.ini và trackers/<tracker>/data/<seq>.txt (MOTChallenge 2D box).

    Mode 3D: TrackEval không có bộ đo 3D nên ghi hình chiếu BEV của box (tâm ± 1 m, nhân bev_scale để thành pixel);
    IoU của hai ô vuông 2 m cạnh nhau xấp xỉ tiêu chí tâm cách ≤ 2 m. Bộ đo trong repo (`evaluate_tracks`) dùng tâm thật.
    """
    out = Path(out)
    gt_root, tr_root = out / "gt", out / "trackers" / tracker_name / "data"
    tr_root.mkdir(parents=True, exist_ok=True)
    seqmap = ["name"]
    for seq, frames in seqs.items():
        d = gt_root / seq / "gt"
        d.mkdir(parents=True, exist_ok=True)
        gid, pid = _ids(frames, 1), _ids(frames, 2)
        glines, plines = [], []
        for t, (_, gt, pr) in enumerate(frames, 1):
            for o in gt:
                x, y, w, h = _mot_box(o.geom, mode, bev_scale)
                glines.append(f"{t},{gid[o.track] + 1},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{0 if o.ignore else 1},1,1")
            for o in pr:
                x, y, w, h = _mot_box(o.geom, mode, bev_scale)
                plines.append(f"{t},{pid[o.track] + 1},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{o.score:.3f},-1,-1,-1")
        (d / "gt.txt").write_text("\n".join(glines) + "\n", encoding="utf-8")
        ini = configparser.ConfigParser()
        ini["Sequence"] = {"name": seq, "imDir": "img1", "frameRate": "2", "seqLength": str(len(frames)),
                           "imWidth": "1600", "imHeight": "900", "imExt": ".jpg"}  # fmt: skip
        with open(gt_root / seq / "seqinfo.ini", "w", encoding="utf-8") as fh:
            ini.write(fh)
        (tr_root / f"{seq}.txt").write_text("\n".join(plines) + "\n", encoding="utf-8")
        seqmap.append(seq)
    (gt_root / "seqmap.txt").write_text("\n".join(seqmap) + "\n", encoding="utf-8")
    return out


def _mot_box(geom, mode: str, bev_scale: float) -> tuple[float, float, float, float]:
    if mode == "2d":
        x1, y1, x2, y2 = geom
        return x1, y1, x2 - x1, y2 - y1
    x, y = geom[:2]
    s = bev_scale
    return (x - 1.0) * s + 5000.0, (y - 1.0) * s + 5000.0, 2.0 * s, 2.0 * s


def run_trackeval(mot_dir: Path, tracker_name: str = "autolabel") -> dict | None:
    """Chạy TrackEval chính thức trên thư mục export_mot (None nếu chưa cài). Trả về HOTA/MOTA/IDF1 gộp."""
    for name, typ in (("float", float), ("int", int), ("bool", bool)):  # TrackEval còn dùng np.float / np.int cũ
        if not hasattr(np, name):
            setattr(np, name, typ)
    try:
        import trackeval
    except ImportError:
        return None
    mot_dir = Path(mot_dir)
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(mot_dir / "gt"), "TRACKERS_FOLDER": str(mot_dir / "trackers"),
                   "TRACKERS_TO_EVAL": [tracker_name], "SEQMAP_FILE": str(mot_dir / "gt" / "seqmap.txt"),
                   "SKIP_SPLIT_FOL": True, "BENCHMARK": "", "SPLIT_TO_EVAL": "", "DO_PREPROC": False,
                   "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt", "PRINT_CONFIG": False})  # fmt: skip
    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                   "PLOT_CURVES": False, "LOG_ON_ERROR": None, "USE_PARALLEL": False, "TIME_PROGRESS": False})  # fmt: skip
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                                                   [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])  # fmt: skip
    r = res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    return {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "DetA": float(np.mean(r["HOTA"]["DetA"])),
            "AssA": float(np.mean(r["HOTA"]["AssA"])), "MOTA": float(r["CLEAR"]["MOTA"]), "IDSW": int(r["CLEAR"]["IDSW"]),
            "IDF1": float(r["Identity"]["IDF1"])}  # fmt: skip
