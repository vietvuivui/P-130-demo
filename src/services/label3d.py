"""Pre-label 3D: đọc dự đoán của một mô hình 3D (file kết quả chuẩn nuScenes), kiểm chứng bằng camera, lưu frame 3D.

    dự đoán (hệ toàn cục) ─► hệ LiDAR của keyframe ─► lọc ngưỡng điểm + phạm vi lớp
                          ─► detector 2D trên 6 camera (YOLOE, có cache) ─► verify3d ─► kết luận + mức rủi ro
                          ─► workspace/frames3d/<mô hình>/<frame_id>.json  (+ point cloud rút gọn cho UI, GT 3D)

Dự đoán lấy từ tools3d/run3d.py (chạy trên máy có GPU): eval/results/det3d/<mô hình>/ui_preds.json.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.models.qa_config import AutoLabelConfig, ClassSpec
from src.models.schemas3d import Box3D, Camera3D, Frame3DRecord, Frame3DSummary, Object3D
from src.services.detectors import DetectorEnsemble
from src.services.geometry import quaternion_to_matrix
from src.services.nuscenes_data import NuScenesMini
from src.services.store import WorkspaceStore
from src.services.verify3d import CLASS_RANGE, Box, CamInput, verify_boxes

log = logging.getLogger(__name__)

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
DISTRACTOR = "__distractor__"
LEVEL_RISK = {"low": 0.1, "medium": 0.5, "high": 0.9}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def verify_config(config: AutoLabelConfig) -> AutoLabelConfig:
    """Cấu hình detector cho bước kiểm chứng: chỉ YOLOE, thêm lớp giả chứa prompt đối thủ (box của lớp đó bị bỏ)."""
    cfg = config.model_copy(deep=True)
    cfg.detection.detectors = ["yoloe"]
    cfg.classes[DISTRACTOR] = ClassSpec(prompts=list(config.verify3d.distractors), height_m=(0, 99), aspect_wh=(0, 99))
    return cfg


def lidar_pose(data: NuScenesMini, lidar_sd: str) -> np.ndarray:
    return data._global_from_ego(lidar_sd) @ data._ego_from_sensor(lidar_sd)


def load_points(data: NuScenesMini, lidar_sd: str, n_sweeps: int) -> np.ndarray:
    """(N, 4) = x, y, z, intensity trong hệ LiDAR của keyframe; gộp thêm các lần quét liền trước nếu có trên đĩa."""
    ref_from_global = np.linalg.inv(lidar_pose(data, lidar_sd))
    parts, tok = [], lidar_sd
    for k in range(n_sweeps + 1):
        sd = data.sample_data.get(tok)
        if sd is None:
            break
        path = data.dataroot / sd["filename"]
        if path.is_file():
            p = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4].astype(np.float64)
            if k:
                p = p[~((np.abs(p[:, 0]) < 1.0) & (np.abs(p[:, 1]) < 1.0))]  # điểm trên nóc xe
                tf = ref_from_global @ lidar_pose(data, tok)
                p[:, :3] = p[:, :3] @ tf[:3, :3].T + tf[:3, 3]
            parts.append(p)
        tok = sd.get("prev")
        if not tok:
            break
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def box_to_lidar(b: dict, lidar_from_global: np.ndarray) -> Box3D:
    c = lidar_from_global[:3, :3] @ np.asarray(b["translation"], float) + lidar_from_global[:3, 3]
    rot = lidar_from_global[:3, :3] @ quaternion_to_matrix(b["rotation"])
    v = lidar_from_global[:3, :3] @ np.array([*b.get("velocity", [0.0, 0.0])[:2], 0.0])
    return Box3D(
        center=[round(float(x), 3) for x in c],
        size=[round(float(x), 3) for x in b["size"]],
        yaw=round(float(math.atan2(rot[1, 0], rot[0, 0])), 4),
        velocity=[round(float(v[0]), 3), round(float(v[1]), 3)],
    )


def gt_boxes(data: NuScenesMini, sample_token: str, lidar_from_global: np.ndarray, category_map: dict) -> list[dict]:
    """Nhãn gốc (nếu có) trong hệ LiDAR, chỉ để hiển thị / đánh giá; không dùng khi kiểm chứng."""
    out = []
    for ann in data._annotations_of_sample.get(sample_token, []):
        label = category_map.get(data.category_of_instance[ann["instance_token"]])
        if label is None:
            continue
        box = box_to_lidar(ann, lidar_from_global)
        out.append(
            {"label": label, **box.model_dump(), "num_pts": ann["num_lidar_pts"] + ann["num_radar_pts"],
             "instance_token": ann["instance_token"]}
        )  # fmt: skip
    return out


def _gray(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))
    except OSError:
        return None


def build_frame(
    data: NuScenesMini,
    ensemble: DetectorEnsemble,
    config: AutoLabelConfig,
    model: str,
    scene: str,
    index: int,
    sample_token: str,
    preds: list[dict],
    min_score: float | None = None,
) -> tuple[Frame3DRecord, np.ndarray, list[dict]]:
    vcfg = config.verify3d
    min_score = vcfg.min_score if min_score is None else min_score
    kf = data._keyframe_data[sample_token]
    lidar_sd = kf["LIDAR_TOP"]
    g_from_l = lidar_pose(data, lidar_sd)
    l_from_g = np.linalg.inv(g_from_l)

    objects: list[Object3D] = []
    for b in sorted(preds, key=lambda b: -b["detection_score"]):
        label, score = b["detection_name"], float(b["detection_score"])
        if score < min_score or label not in CLASS_RANGE:
            continue
        box = box_to_lidar(b, l_from_g)
        if math.hypot(box.center[0], box.center[1]) > CLASS_RANGE[label]:
            continue
        objects.append(Object3D(object_id=str(len(objects) + 1), label=label, score=round(score, 4), box=box))

    cams, inputs = {}, []
    images = [(kf[c], data.dataroot / data.sample_data[kf[c]]["filename"]) for c in CAMERAS if c in kf]
    dets_all = ensemble.detect_batch(images) if objects else [[] for _ in images]
    for (sd_tok, path), dets, cam in zip(images, dets_all, [c for c in CAMERAS if c in kf], strict=True):
        sd = data.sample_data[sd_tok]
        cs = data.calibrated_sensor[sd["calibrated_sensor_token"]]
        intr = np.asarray(cs["camera_intrinsic"], float)
        c_from_l = data.cam_from_global(sd_tok) @ g_from_l
        cams[cam] = Camera3D(
            sd_token=sd_tok, path=sd["filename"], width=sd["width"], height=sd["height"], intrinsic=intr.tolist(),
            cam_from_lidar=np.round(c_from_l, 6).tolist(),
        )  # fmt: skip
        keep = [
            {"bbox": d.bbox, "label": d.label, "score": d.score, "prompt": d.label}
            for d in dets
            if d.label != DISTRACTOR and d.score >= vcfg.det_conf
        ]
        inputs.append(CamInput(cam, intr, c_from_l, sd["width"], sd["height"], _gray(path) if objects else None, keep))

    pts = load_points(data, lidar_sd, vcfg.lidar_sweeps)
    boxes = [Box(o.label, o.score, np.array(o.box.center), np.array(o.box.size), o.box.yaw) for o in objects]
    for o, v in zip(objects, verify_boxes(boxes, inputs, pts), strict=True):
        o.verify = v
    record = Frame3DRecord(
        frame_id=f"{scene}_{index:03d}",
        model=model,
        sample_token=sample_token,
        scene=scene,
        index=index,
        lidar_sd_token=lidar_sd,
        global_from_lidar=np.round(g_from_l, 6).tolist(),
        cameras=cams,
        objects=objects,
        frame_risk=max((LEVEL_RISK[o.verify.level] for o in objects), default=0.0),
        created_at=now_iso(),
    )
    return record, pts, gt_boxes(data, sample_token, l_from_g, config.gt_category_map)


def best_threshold(pr: dict) -> float:
    """Ngưỡng điểm cho F1 cao nhất trên bảng precision/recall theo ngưỡng (metrics.json của tools3d/run3d.py).

    Điểm tin cậy của mỗi mô hình có thang khác nhau (PGD hầu như không có box >= 0.4, CenterPoint thì nhiều), nên một
    ngưỡng chung 0.3 sẽ bỏ gần hết box của mô hình này và giữ quá nhiều box của mô hình kia."""
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(pr["precision"], pr["recall"], strict=True)]
    return float(pr["thresholds"][int(np.argmax(f1))])


def save_points(store: WorkspaceStore, frame_id: str, pts: np.ndarray, max_points: int) -> None:
    """Point cloud keyframe (chỉ lần quét chính, không gộp sweep) rút gọn, float32 x y z intensity."""
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    path = store.points_path(frame_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts[:, :4].astype(np.float32).tofile(path)


def run_label3d(
    store: WorkspaceStore,
    data: NuScenesMini,
    config: AutoLabelConfig,
    model: str,
    predictions: dict[str, list[dict]],
    scenes: list[str] | None = None,
    overwrite: bool = False,
    min_score: float | None = None,
) -> int:
    """Tạo frame 3D cho mọi keyframe có dự đoán. Frame đã có (người có thể đã duyệt) giữ nguyên trừ khi overwrite."""
    ensemble = DetectorEnsemble(verify_config(config), store.root / "cache" / "detections")
    done = 0
    keys = [k for k in data.keyframes(scenes) if k[2] in predictions]
    for n, (scene, index, token) in enumerate(keys):
        fid = f"{scene}_{index:03d}"
        if not overwrite and store.load_frame3d(model, fid) is not None:
            continue
        record, pts, gt = build_frame(data, ensemble, config, model, scene, index, token, predictions[token], min_score)
        store.save_frame3d(record)
        kf_pts = load_points(data, record.lidar_sd_token, 0)
        save_points(store, fid, kf_pts if len(kf_pts) else pts, config.verify3d.max_points_ui)
        store.save_aux("gt3d", fid, gt)
        done += 1
        counts = {lv: sum(o.verify.level == lv for o in record.objects) for lv in ("low", "medium", "high")}
        log.info("[%d/%d] %s %s: %d box 3D %s", n + 1, len(keys), model, fid, len(record.objects), counts)
    return done


def summarize3d(f: Frame3DRecord) -> Frame3DSummary:
    counts = {
        lv: sum(o.verify is not None and o.verify.level == lv for o in f.objects) for lv in ("low", "medium", "high")
    }
    return Frame3DSummary(
        frame_id=f.frame_id, scene=f.scene, index=f.index, status=f.status, frame_risk=f.frame_risk, counts=counts,
        pending=sum(o.review.status == "pending" for o in f.objects), n_objects=len(f.objects),
    )  # fmt: skip
