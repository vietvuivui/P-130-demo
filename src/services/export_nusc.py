"""Xuất nhãn 3D của dự án thành dataset nuScenes: bảng sample_annotation + instance đúng chuẩn nuScenes (hệ toàn cục),
kèm các bảng gốc của dữ liệu, file kết quả dạng detection và log sửa. Đóng thành một file zip để tải về.

- Chỉ lấy keyframe đã approve (mặc định) hoặc cả nhãn chưa duyệt (include_pending=True), bỏ box đã xoá.
- Box cùng track (tinh chỉnh theo track khi dự đoán) thành cùng một instance; box người vẽ thêm là instance riêng.
"""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.schemas3d import Frame3DRecord, Object3D
from src.services.ingest.nusc_writer import quat_from_matrix
from src.services.review3d import read_log
from src.services.store import WorkspaceStore

# lớp nội bộ -> category nuScenes (chọn category phổ biến nhất của lớp)
CLASS_TO_CATEGORY = {
    "car": "vehicle.car", "truck": "vehicle.truck", "bus": "vehicle.bus.rigid", "trailer": "vehicle.trailer",
    "construction_vehicle": "vehicle.construction", "pedestrian": "human.pedestrian.adult",
    "motorcycle": "vehicle.motorcycle", "bicycle": "vehicle.bicycle", "traffic_cone": "movable_object.trafficcone",
    "barrier": "movable_object.barrier",
}  # fmt: skip


def _global_box(f: Frame3DRecord, o: Object3D) -> tuple[list, list, list, list]:
    g = np.asarray(f.global_from_lidar)
    c = g[:3, :3] @ np.asarray(o.box.center) + g[:3, 3]
    yaw = o.box.yaw
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    v = g[:3, :3] @ np.array([*o.box.velocity, 0.0])
    return (
        c.round(3).tolist(),
        [round(float(s), 3) for s in o.box.size],
        quat_from_matrix(g[:3, :3] @ rz),
        v[:2].round(3).tolist(),
    )


def export_nuscenes(store: WorkspaceStore, dataroot: Path, version: str, model: str, out_zip: Path,
                    include_pending: bool = False) -> dict:  # fmt: skip
    table_dir = Path(dataroot) / version
    categories = {c["name"]: c["token"] for c in json.loads((table_dir / "category.json").read_text(encoding="utf-8"))}
    sample_ts = {
        s["token"]: s["timestamp"] for s in json.loads((table_dir / "sample.json").read_text(encoding="utf-8"))
    }
    frames = [f for f in store.list_frames3d(model) if include_pending or f.status == "approved"]
    if not frames:
        raise ValueError("Chưa có keyframe 3D nào được approve" if not include_pending else "Chưa có frame 3D nào")

    anns: list[dict] = []
    by_instance: dict[str, list[dict]] = defaultdict(list)
    results: dict[str, list[dict]] = {}
    for f in sorted(frames, key=lambda f: sample_ts.get(f.sample_token, 0)):
        rows = []
        for o in f.objects:
            keep = o.review.status == "approved" or (include_pending and o.review.status == "pending")
            if not keep:
                continue
            label = o.review.final_label or o.label
            cat = categories.get(CLASS_TO_CATEGORY.get(label, ""), None)
            if cat is None:
                continue
            tr, size, rot, vel = _global_box(f, o)
            inst_key = (
                f"{f.scene}/t{o.track_id}" if o.track_id and o.source == "model" else f"{f.frame_id}/o{o.object_id}"
            )
            ann = dict(
                token=uuid.uuid4().hex, sample_token=f.sample_token, instance_token="", visibility_token="",
                attribute_tokens=[], translation=tr, size=size, rotation=rot, prev="", next="",
                num_lidar_pts=int(o.verify.lidar_points) if o.verify else 0, num_radar_pts=0,
            )  # fmt: skip
            anns.append(ann)
            by_instance[inst_key].append((ann, cat))
            rows.append(dict(sample_token=f.sample_token, translation=tr, size=size, rotation=rot, velocity=vel,
                             detection_name=label, detection_score=round(o.score, 4), attribute_name="",
                             review_status=o.review.status, review_action=o.review.action, source=o.source))  # fmt: skip
        results[f.sample_token] = rows
    instances = []
    for items in by_instance.values():
        tok = uuid.uuid4().hex
        for i, (ann, _) in enumerate(items):
            ann["instance_token"] = tok
            ann["prev"] = items[i - 1][0]["token"] if i else ""
            ann["next"] = items[i + 1][0]["token"] if i + 1 < len(items) else ""
        instances.append(dict(token=tok, category_token=items[0][1], nbr_annotations=len(items),
                              first_annotation_token=items[0][0]["token"], last_annotation_token=items[-1][0]["token"]))  # fmt: skip

    stage = out_zip.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    tdir = stage / version
    tdir.mkdir(parents=True)
    for t in table_dir.glob("*.json"):
        shutil.copy2(t, tdir / t.name)
    for name in ("sample_annotation", "instance"):
        orig = tdir / f"{name}.json"
        if orig.exists() and json.loads(orig.read_text(encoding="utf-8")):
            orig.rename(tdir / f"{name}.original.json")  # nhãn gốc đi kèm dữ liệu (nếu có) được giữ lại
    (tdir / "sample_annotation.json").write_text(json.dumps(anns), encoding="utf-8")
    (tdir / "instance.json").write_text(json.dumps(instances), encoding="utf-8")
    meta = dict(use_camera=True, use_lidar=True, use_radar=False, use_map=False, use_external=False)
    (stage / "labels3d_nusc.json").write_text(json.dumps(dict(meta=meta, results=results)), encoding="utf-8")
    (stage / "corrections3d.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in read_log(store, model)), encoding="utf-8"
    )
    (stage / "README.txt").write_text(
        f"Nhãn 3D xuất từ AutoLabel 3D.\n\n{version}/            bảng nuScenes của dữ liệu, sample_annotation.json + "
        "instance.json là nhãn đã duyệt\n                         (nhãn gốc nếu có: *.original.json). Chép thư mục này "
        "đè lên bảng của dữ liệu gốc\n                         (cùng samples/ sweeps/) là dùng được với nuscenes-devkit / "
        "MMDetection3D.\nlabels3d_nusc.json       cùng nhãn ở định dạng kết quả detection của nuScenes\n"
        "corrections3d.jsonl      log thao tác của người duyệt\n",
        encoding="utf-8",
    )
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in stage.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(stage))
    shutil.rmtree(stage)
    return dict(file=out_zip.name, frames=len(frames), annotations=len(anns), instances=len(instances))
