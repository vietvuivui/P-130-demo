"""Bước 6: xuất tập nhãn 2D đã được người duyệt (COCO + JSONL + correction log + báo cáo QA).

Chỉ frame đã approve VÀ nhãn cuối không còn lỗi QC chưa xử lý mới được xuất. manifest.json ghi trạng thái phát hành
(READY / NOT_READY theo checklist QC), nguồn gốc nhãn, SHA256 từng file, băm config và commit để tái lập.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import ExportResponse
from src.services.qc.report import collect, dataset_report, render_qa_report
from src.services.review import now_iso
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore

EXPORT_FILES = ("coco.json", "labels.jsonl", "corrections.jsonl", "qc_log.jsonl", "qa_report.md", "manifest.json")
REPO_ROOT = Path(__file__).resolve().parents[2]


class NothingToExportError(ValueError):
    pass


class NotReadyError(ValueError):
    pass


def valid_mask(obj) -> list[float] | None:
    """Mask còn dùng được: có mask, object không bị xoá, người không sửa box / không vẽ lại."""
    if not obj.mask or obj.review.status == "deleted" or obj.review.action in ("EDIT_BOX", "ADD_BOX"):
        return None
    return obj.mask


def export_dataset(store: WorkspaceStore, config: AutoLabelConfig, require_ready: bool = False) -> ExportResponse:
    qc = collect(store, config)
    # FR-18: chỉ frame đã approve mới được xuất
    if not qc.approved:
        waiting = sum(f.status == "editing" for f in qc.frames)
        raise NothingToExportError(f"Chưa có frame nào được approve ({waiting} frame đang sửa)")
    report = dataset_report(store, config, qc)
    if require_ready and report["status"] != "READY":
        failed = "; ".join(f"{c['label']}: {c['detail']}" for c in report["checks"] if not c["ok"])
        raise NotReadyError(f"Dataset chưa READY — {failed}")
    open_by_frame = qc.open_by_frame
    approved = [f for f in qc.approved if f.frame_id not in open_by_frame]
    if not approved:
        raise NothingToExportError(f"Cả {len(qc.approved)} frame đã approve đều còn lỗi QC chưa xử lý (xem tab QC)")

    export_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = store.exports_dir / export_id
    out.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(config.classes)]
    cat_id = {c["name"]: c["id"] for c in categories}
    images, annotations, lines = [], [], []
    sources, actions = Counter(), Counter()

    for img_id, frame in enumerate(approved, start=1):
        # Warning QC người đã xác nhận "đã kiểm, giữ nguyên" trên từng object: ghi kèm để người dùng dataset biết
        acked: dict[str, list[str]] = {}
        for f in qc.findings[frame.frame_id]:
            if f.acked and f.object_id:
                acked.setdefault(f.object_id, []).append(f.code)
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
            sources[obj.source] += 1
            actions[obj.review.action] += 1
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
                "qc_acknowledged": acked.get(obj.object_id, []),
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
                "width": frame.image.width,
                "height": frame.image.height,
                "objects": objs,
            }
        )

    exported = {f.frame_id for f in approved}
    corrections = [e for e in store.corrections() if e["frame_id"] in exported]
    qc_log = [e for e in store.qc_events() if e.get("frame_id") in exported or e.get("frame_id") is None]

    (out / "coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out / "labels.jsonl", lines)
    _write_jsonl(out / "corrections.jsonl", corrections)
    _write_jsonl(out / "qc_log.jsonl", qc_log)
    manifest = {
        "export_id": export_id,
        "created_at": now_iso(),
        "release_status": report["status"],
        "n_frames": len(approved),
        "n_objects": len(annotations),
        "frames_skipped_not_approved": len(qc.frames) - len(qc.approved),
        "frames_skipped_qc": sorted(open_by_frame),
        "classes": list(config.classes),
        "detectors": sorted({d for f in approved for d in f.detectors}),
        "risk_weights": config.qa.risk.weights.model_dump(),
        "provenance": {"source": dict(sources.most_common()), "human_action": dict(actions.most_common())},
        "qc": {
            "checks": report["checks"],
            "audit": report["audit"],
            "acked_findings": report["n_acked_findings"],
        },
        "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
        "git_commit": _git_commit(),
    }
    (out / "qa_report.md").write_text(render_qa_report(report, manifest), encoding="utf-8")
    # Băm mọi file (trừ chính manifest) để người nhận kiểm được dataset không bị sửa sau khi xuất
    manifest["files_sha256"] = {name: _sha256(out / name) for name in EXPORT_FILES if name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExportResponse(
        export_id=export_id,
        n_frames=len(approved),
        n_objects=len(annotations),
        files=list(EXPORT_FILES),
        release_status=report["status"],
        frames_skipped_qc=sorted(open_by_frame),
    )


def _write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    """Commit của code lúc xuất; hậu tố -dirty khi còn thay đổi chưa commit (khi đó hash không đủ để tái lập)."""
    try:
        res = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None
