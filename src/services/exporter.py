"""Bước 6: xuất tập nhãn 2D đã được người duyệt (COCO + JSONL + correction log)."""

from __future__ import annotations

import json
from datetime import datetime

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import ExportResponse
from src.services.review import now_iso
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore

EXPORT_FILES = ("coco.json", "labels.jsonl", "corrections.jsonl", "manifest.json")


class NothingToExportError(ValueError):
    pass


def valid_mask(obj) -> list[float] | None:
    """Mask còn dùng được: có mask, object không bị xoá, người không sửa box / không vẽ lại."""
    if not obj.mask or obj.review.status == "deleted" or obj.review.action in ("EDIT_BOX", "ADD_BOX"):
        return None
    return obj.mask


def export_dataset(store: WorkspaceStore, config: AutoLabelConfig) -> ExportResponse:
    frames = store.list_frames()
    # FR-18: chỉ frame đã approve mới được xuất
    approved = [f for f in frames if f.status == "approved"]
    if not approved:
        waiting = sum(f.status == "editing" for f in frames)
        raise NothingToExportError(f"Chưa có frame nào được approve ({waiting} frame đang sửa)")

    export_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = store.exports_dir / export_id
    out.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(config.classes)]
    cat_id = {c["name"]: c["id"] for c in categories}
    images, annotations, lines = [], [], []

    for img_id, frame in enumerate(approved, start=1):
        images.append(
            {
                "id": img_id,
                # Ảnh nuScenes: đường dẫn tương đối so với dataroot; video tải lên: tương đối so với workspace
                "file_name": frame.image.path.removeprefix(WORKSPACE_PREFIX),
                "image_root": "workspace" if frame.image.path.startswith(WORKSPACE_PREFIX) else "nuscenes",
                "width": frame.image.width,
                "height": frame.image.height,
                "frame_id": frame.frame_id,
                "sample_token": frame.sample_token,
                "sample_data_token": frame.image.sd_token,
            }
        )
        objs = []
        for obj in frame.objects:
            if obj.review.status != "approved":
                continue
            x1, y1, x2, y2 = obj.review.final_bbox
            record = {
                "object_id": obj.object_id,
                # Cùng một object ở các frame khác nhau có cùng track_id (nhờ lan truyền)
                "track_id": obj.track_id,
                "propagated_from": obj.propagation.keyframe_id if obj.propagation else None,
                "bbox": obj.review.final_bbox,
                "class": obj.review.final_label,
                "source": obj.source,
                "model_score": obj.score if obj.source != "human" else None,
                "risk": obj.qa.risk if obj.qa else None,
                "issues": [i.code for i in obj.qa.issues] if obj.qa else [],
                "human_action": obj.review.action,
                "reviewer": obj.review.reviewer,
            }
            objs.append(record)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": img_id,
                    "category_id": cat_id[obj.review.final_label],
                    "bbox": [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
                    "area": round((x2 - x1) * (y2 - y1), 1),
                    "iscrowd": 0,
                    # mask sơ bộ của model (FR-04); rỗng khi không có hoặc người đã sửa box (mask không còn khớp)
                    "segmentation": [valid_mask(obj)] if valid_mask(obj) else [],
                    **{k: v for k, v in record.items() if k not in ("bbox", "class")},
                }
            )
        lines.append(
            {
                "frame_id": frame.frame_id,
                "sample_token": frame.sample_token,
                "image": frame.image.path.removeprefix(WORKSPACE_PREFIX),
                "objects": objs,
            }
        )

    exported = {f.frame_id for f in approved}
    corrections = [e for e in store.corrections() if e["frame_id"] in exported]

    (out / "coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out / "labels.jsonl", lines)
    _write_jsonl(out / "corrections.jsonl", corrections)
    manifest = {
        "export_id": export_id,
        "created_at": now_iso(),
        "n_frames": len(approved),
        "n_objects": len(annotations),
        "frames_skipped_not_approved": len(frames) - len(approved),
        "classes": list(config.classes),
        "detectors": sorted({d for f in approved for d in f.detectors}),
        "risk_weights": config.qa.risk.weights.model_dump(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExportResponse(
        export_id=export_id, n_frames=len(approved), n_objects=len(annotations), files=list(EXPORT_FILES)
    )


def _write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
