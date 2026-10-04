"""API phần 3D: danh sách frame 3D theo mô hình, point cloud, ảnh 6 camera, thao tác duyệt, số liệu, so sánh mô hình."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, Response

from src.api.routes import _error, data_source, get_config, get_dataroot, get_store, undo_redo
from src.config import get_settings
from src.models.qa_config import AutoLabelConfig
from src.models.schemas import RejectRequest, ReviewerRequest
from src.models.schemas3d import Action3DRequest, Frame3DRecord, Frame3DSummary
from src.services import bev, history, review, review3d
from src.services.label3d import summarize3d
from src.services.store import WorkspaceStore

router3d = APIRouter(prefix="/3d")
DET3D_DIR = Path("eval/results/det3d")


def _load(store: WorkspaceStore, model: str, frame_id: str) -> Frame3DRecord:
    try:
        frame = store.load_frame3d(model, frame_id)
    except ValueError as e:
        raise _error(400, "INVALID_ID", str(e)) from e
    if frame is None:
        raise _error(404, "FRAME_NOT_FOUND", f"Không có frame 3D {frame_id} của {model}")
    return frame


def _raw_models(store: WorkspaceStore) -> dict[str, Path]:
    """Mô hình LiDAR đơn lẻ đã có dự đoán trong work3d của dự án (bước Dự đoán 3D chạy từng mô hình rồi mới gộp)."""
    d = Path(store.root).parent / "work3d" / "preds"
    if not d.is_dir():
        return {}
    files = {p.name: p / "pred_instances_3d" / "results_nusc.json" for p in sorted(d.iterdir()) if p.is_dir()}
    return {m: f for m, f in files.items() if f.exists()}


@router3d.get("/models")
def models(store: WorkspaceStore = Depends(get_store)) -> list[dict]:
    """Mô hình 3D chọn được trên UI, kèm số liệu so sánh (nếu đã chạy tools3d/run3d.py eval).

    built = đã có frame 3D trong workspace. Mô hình đơn lẻ của dự án mới chỉ có dự đoán (built = false) được tạo frame
    khi người dùng chọn lần đầu: POST /3d/models/{model}/build."""
    summary = _det3d_summary().get("models", {})
    out = []
    built = store.models3d()
    for m in built:
        frames = store.list_frames3d(m)
        s = summary.get(m, {})
        out.append({
            "model": m, "label": s.get("label", m), "sensor": s.get("sensor"), "frames": len(frames),
            "approved": sum(f.status == "approved" for f in frames), "mAP": s.get("mAP"), "NDS": s.get("NDS"),
            "built": True,
        })  # fmt: skip
    for m in _raw_models(store):
        if m not in built:
            s = summary.get(m, {})
            out.append({"model": m, "label": s.get("label", m), "sensor": s.get("sensor", "LiDAR"), "frames": 0,
                        "approved": 0, "mAP": s.get("mAP"), "NDS": s.get("NDS"), "built": False})  # fmt: skip
    # Bản gộp (ensemble) của dự án đứng đầu: UI mở nó mặc định; còn lại theo NDS giảm dần
    return sorted(out, key=lambda r: (r["model"] != "ensemble", -(r["NDS"] or 0), r["model"]))


def _build_key(store: WorkspaceStore) -> str:
    return f"{Path(store.root).resolve()}:build3d"


@router3d.get("/models/build")
def build_status(store: WorkspaceStore = Depends(get_store)):
    from src.services import jobs

    return jobs.status(_build_key(store))


@router3d.post("/models/{model}/build")
def build_model(model: str, request: Request, store: WorkspaceStore = Depends(get_store),
                config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    """Tạo frame 3D (kiểm chứng bằng camera) cho một mô hình đơn lẻ từ dự đoán đã có của dự án. Chạy nền."""
    from src.services import jobs

    raw = _raw_models(store)
    if model not in raw:
        raise _error(404, "MODEL_NOT_FOUND", f"Dự án không có dự đoán của mô hình {model}")
    root, version = data_source(request)

    def work(progress):
        from src.services.label3d import auto_min_score, run_label3d
        from src.services.nuscenes_data import NuScenesMini

        progress(0.02, f"{model}: đọc dự đoán…")
        preds = json.loads(raw[model].read_text(encoding="utf-8"))["results"]
        min_score = auto_min_score(model, config.verify3d.min_score, DET3D_DIR)
        n = run_label3d(store, NuScenesMini(root, version), config, model, preds, min_score=min_score,
                        progress=lambda i, total: progress(i / max(total, 1), f"{model}: {i}/{total} keyframe"))  # fmt: skip
        return {"model": model, "created": n, "frames": len(store.frame3d_ids(model)), "min_score": min_score}

    return jobs.start(_build_key(store), work)


@router3d.get("/frames", response_model=list[Frame3DSummary])
def list_frames(model: str, store: WorkspaceStore = Depends(get_store)):
    try:
        return [summarize3d(f) for f in store.list_frames3d(model)]
    except ValueError as e:
        raise _error(400, "INVALID_ID", str(e)) from e


@router3d.get("/frames/{model}/{frame_id}", response_model=Frame3DRecord)
def get_frame(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    return _load(store, model, frame_id)


@router3d.get("/frames/{model}/{frame_id}/points")
def points(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    """Point cloud rút gọn: float32 liên tiếp x, y, z, intensity (hệ LiDAR của keyframe)."""
    _load(store, model, frame_id)
    path = store.points_path(frame_id)
    if not path.exists():
        raise _error(404, "NO_POINTS", "Chưa có point cloud của frame này")
    return Response(path.read_bytes(), media_type="application/octet-stream")


@router3d.get("/frames/{model}/{frame_id}/gt")
def gt(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, model, frame_id)
    return store.load_aux("gt3d", frame_id) or []


@router3d.get("/frames/{model}/{frame_id}/image/{camera}")
def image(model: str, frame_id: str, camera: str, store: WorkspaceStore = Depends(get_store),
          dataroot: Path = Depends(get_dataroot), config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    frame = _load(store, model, frame_id)
    cam = frame.cameras.get(camera)
    if cam is None:
        raise _error(404, "NO_CAMERA", f"Frame không có {camera}")
    path = dataroot / cam.path
    if not path.is_file():
        raise _error(404, "IMAGE_NOT_FOUND", f"Không thấy ảnh {cam.path} (kiểm tra NUSCENES_DATAROOT)")
    from src.services.privacy import anonymized_path

    path = anonymized_path(store.root, path, config.privacy)  # làm mờ mặt / biển số (FR-03)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


def _bev_neighbors(store: WorkspaceStore, model: str, frame: Frame3DRecord, dataroot: Path) -> list:
    """Keyframe ±BEV_NEIGHBORS cùng scene (id `<scene>_<index:03d>`), gần nhất trước: (ảnh 6 camera, frame, LiDAR)."""
    out = []
    for d in sorted(range(-bev.BEV_NEIGHBORS, bev.BEV_NEIGHBORS + 1), key=abs):
        if d == 0 or frame.index + d < 0:
            continue
        nid = f"{frame.scene}_{frame.index + d:03d}"
        try:
            nb = store.load_frame3d(model, nid)
        except ValueError:
            continue
        if nb is None or nb.scene != frame.scene:
            continue
        p = store.points_path(nid)
        pts = np.fromfile(p, np.float32).reshape(-1, 4) if p.exists() else None
        imgs = {c: bev.load_rgb(dataroot / cam.path) for c, cam in nb.cameras.items()}
        if any(v is not None for v in imgs.values()):
            out.append((imgs, nb, pts))
    return out


@router3d.get("/frames/{model}/{frame_id}/bev")
def bev_image(model: str, frame_id: str, range_m: float = Query(40.0, alias="range", ge=10, le=80),
              res: float = Query(0.1, ge=0.05, le=0.5), fuse: bool = True, store: WorkspaceStore = Depends(get_store),
              dataroot: Path = Depends(get_dataroot)):  # fmt: skip
    """Ảnh BEV ghép 6 camera bằng homography mặt đường (PNG RGBA). Header X-Ground-Z: độ cao mặt đường (hệ LiDAR).
    `fuse`: ghép thêm mặt đường từ các keyframe lân cận của cùng scene (lấp vùng bị xe che, ảnh xa hết nhoè)."""
    frame = _load(store, model, frame_id)
    pts_path = store.points_path(frame_id)
    pts = np.fromfile(pts_path, np.float32).reshape(-1, 4) if pts_path.exists() else None
    z0 = bev.ground_height(pts)
    # camera và mặt đường như nhau giữa các mô hình: cache theo frame
    tag = "" if fuse else "_single"
    cache = store.root / "bev3d" / f"{store.check_id(frame_id)}_{range_m:g}_{res:g}{tag}_v{bev.BEV_VERSION}.png"
    if cache.exists():
        data = cache.read_bytes()
    else:
        images = {c: bev.load_rgb(dataroot / cam.path) for c, cam in frame.cameras.items()}
        if not any(v is not None for v in images.values()):
            raise _error(404, "IMAGE_NOT_FOUND", "Không đọc được ảnh camera nào (kiểm tra NUSCENES_DATAROOT)")
        nbrs = _bev_neighbors(store, model, frame, dataroot) if fuse else None
        data = bev.to_png(bev.bev_mosaic(images, frame, z0, range_m, res, points=pts, neighbors=nbrs))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    headers = {"X-Ground-Z": str(z0), "X-BEV-Range": f"{range_m:g}", "Cache-Control": "max-age=3600"}
    return Response(data, media_type="image/png", headers=headers)


@router3d.post("/frames/{model}/{frame_id}/actions", response_model=Frame3DRecord)
def action(model: str, frame_id: str, req: Action3DRequest, store: WorkspaceStore = Depends(get_store),
           config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    frame = _load(store, model, frame_id)
    before = frame.model_dump_json()
    try:
        entry = review3d.apply_action(frame, req, req.reviewer or get_settings().reviewer_name, list(config.classes))
    except review3d.Review3DError as e:
        raise _error(422, "INVALID_ACTION", str(e)) from e
    store.save_frame3d(frame)
    review3d.append_log(store, [entry])
    history.record(store, history.kind_3d(model), frame_id, before)
    return frame


@router3d.post("/frames/{model}/{frame_id}/approve-low-risk", response_model=Frame3DRecord)
def approve_low(model: str, frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    before = frame.model_dump_json()
    try:
        entries = review3d.approve_low_risk(frame, req.reviewer or get_settings().reviewer_name)
    except review3d.Review3DError as e:
        raise _error(409, "FRAME_APPROVED", str(e)) from e
    store.save_frame3d(frame)
    review3d.append_log(store, entries)
    if entries:
        history.record(store, history.kind_3d(model), frame_id, before)
    return frame


@router3d.post("/frames/{model}/{frame_id}/approve", response_model=Frame3DRecord)
def approve(model: str, frame_id: str, req: ReviewerRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        review3d.approve_frame(frame, reviewer, req.review_time_s)
    except review3d.Review3DError as e:
        raise _error(409, "PENDING_OBJECTS", str(e)) from e
    store.save_frame3d(frame)
    store.append_event(dict(type="approve", mode="3d", model=model, frame_id=frame_id, reviewer=reviewer,
                            review_time_s=req.review_time_s, n_objects=len(frame.objects)))  # fmt: skip
    return frame


@router3d.post("/frames/{model}/{frame_id}/propagate")
def propagate(model: str, frame_id: str, request: Request, store: WorkspaceStore = Depends(get_store),
              config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    """Mang box người đã duyệt ở keyframe này sang các keyframe sau còn "auto" (src/services/propagation3d.py)."""
    from src.api.routes import _nuscenes
    from src.services.propagation3d import propagate3d_from

    _load(store, model, frame_id)
    timestamps = None
    try:
        data = _nuscenes(*data_source(request))
        timestamps = lambda tok: data.sample[tok]["timestamp"]  # noqa: E731
    except (FileNotFoundError, KeyError):
        pass  # không có bảng nuScenes: coi hai keyframe cách nhau 0.5 s
    try:
        resp = propagate3d_from(store, model, frame_id, config.propagation3d, timestamps)
    except ValueError as e:
        raise _error(409, "NOT_APPROVED", str(e)) from e
    except KeyError:
        resp = propagate3d_from(store, model, frame_id, config.propagation3d, None)
    store.append_event(dict(type="propagate", mode="3d", model=model, frame_id=frame_id,
                            frames=len(resp["frames_updated"]), objects=resp["objects_propagated"]))  # fmt: skip
    return resp


@router3d.post("/frames/{model}/{frame_id}/reject", response_model=Frame3DRecord)
def reject(model: str, frame_id: str, req: RejectRequest, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name
    try:
        review.reject_frame(frame, reviewer, req.reason)
    except review.ReviewError as e:
        raise _error(422, "REASON_REQUIRED", str(e)) from e
    store.save_frame3d(frame)
    store.append_event(dict(type="reject", mode="3d", model=model, frame_id=frame_id, reviewer=reviewer,
                            reason=frame.reject_reason))  # fmt: skip
    return frame


@router3d.post("/frames/{model}/{frame_id}/undo", response_model=Frame3DRecord)
@router3d.post("/frames/{model}/{frame_id}/redo", response_model=Frame3DRecord)
def undo(model: str, frame_id: str, req: ReviewerRequest, request: Request, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    reviewer = req.reviewer or get_settings().reviewer_name

    def log(objs, restored, action, who):
        review3d.append_log(
            store, [dict(review3d._entry(restored, o, action, who), timestamp=review.now_iso()) for o in objs]
        )

    direction = "redo" if request.url.path.endswith("/redo") else "undo"
    return undo_redo(store, history.kind_3d(model), frame, direction, reviewer, Frame3DRecord, store.save_frame3d, log)


@router3d.get("/frames/{model}/{frame_id}/history")
def frame_history(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    _load(store, model, frame_id)
    return history.counts(store, history.kind_3d(model), frame_id)


@router3d.post("/frames/{model}/{frame_id}/reopen", response_model=Frame3DRecord)
def reopen(model: str, frame_id: str, store: WorkspaceStore = Depends(get_store)):
    frame = _load(store, model, frame_id)
    review3d.reopen(frame)
    store.save_frame3d(frame)
    store.append_event(dict(type="reopen", mode="3d", model=model, frame_id=frame_id,
                            reviewer=get_settings().reviewer_name))  # fmt: skip
    return frame


@router3d.get("/report.csv")
def report_csv(
    model: str, kind: str = Query("frames", pattern="^(frames|summary)$"), store: WorkspaceStore = Depends(get_store)
):
    from src.services import productivity

    body = (
        productivity.frames_csv(store.list_frames3d(model), "3d")
        if kind == "frames"
        else productivity.summary_csv(metrics(model, store))
    )
    return Response(
        "\ufeff" + body, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="autolabel3d_{model}_{kind}.csv"'},
    )  # fmt: skip


@router3d.get("/metrics")
def metrics(model: str, store: WorkspaceStore = Depends(get_store)):
    from src.services import productivity

    frames = store.list_frames3d(model)
    m = review3d.metrics3d(frames)
    events = [e for e in store.events() if e.get("model") in (None, model)]
    m["productivity"] = {
        "reviewers": productivity.reviewer_stats(frames, events, review3d.read_log(store, model), "3d"),
        "inference": productivity.inference_stats(frames),
    }
    return m


@router3d.get("/corrections")
def corrections(model: str | None = None, limit: int = Query(500, le=5000), store: WorkspaceStore = Depends(get_store)):
    return review3d.read_log(store, model)[-limit:]


@router3d.post("/export")
def export(model: str, store: WorkspaceStore = Depends(get_store)):
    from src.services.label3d import now_iso

    out = store.exports_dir / f"3d-{model}-{now_iso().replace(':', '').replace('-', '')[:15]}"
    try:
        return review3d.export3d(store, model, out)
    except review3d.Review3DError as e:
        raise _error(409, "NOTHING_TO_EXPORT", str(e)) from e


@router3d.post("/export-kitti")
def export_kitti(model: str, request: Request, camera: str = "CAM_FRONT", include_pending: bool = False,
                 store: WorkspaceStore = Depends(get_store), dataroot=Depends(get_dataroot),
                 config: AutoLabelConfig = Depends(get_config)):  # fmt: skip
    """Box 3D đã duyệt -> định dạng KITTI object (zip), FR-17."""
    from src.services import export_kitti as ek
    from src.services.label3d import now_iso
    from src.services.privacy import image_loader

    name = f"kitti-{model}-{now_iso().replace(':', '').replace('-', '')[:15]}.zip"
    try:
        info = ek.export_kitti(store, Path(dataroot), data_source(request)[1], model, store.exports_dir / name,
                               camera=camera, include_pending=include_pending,
                               image_loader=image_loader(store, config.privacy))  # fmt: skip
    except ValueError as e:
        raise _error(409, "NOTHING_TO_EXPORT", str(e)) from e
    return dict(info, url=f"{request.url.path.rsplit('/export-kitti', 1)[0]}/exports/{name}")


@router3d.get("/exports/{name}")
def download_export(name: str, store: WorkspaceStore = Depends(get_store)):
    if not name.endswith(".zip") or "/" in name or "\\" in name or name.startswith("."):
        raise _error(404, "FILE_NOT_FOUND", name)
    path = store.exports_dir / name
    if not path.exists():
        raise _error(404, "FILE_NOT_FOUND", name)
    return FileResponse(path, filename=name, media_type="application/zip")


def _det3d_summary() -> dict:
    path = DET3D_DIR / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router3d.get("/compare")
def compare():
    """Bảng so sánh mô hình 3D (mAP/NDS chuẩn nuScenes trên scene val, tốc độ, kết quả kiểm chứng)."""
    out = _det3d_summary()
    for name in out.get("models", {}):
        v = DET3D_DIR / name / "verify_eval.json"
        if v.exists():
            out["models"][name]["verify"] = json.loads(v.read_text(encoding="utf-8"))
    return out
