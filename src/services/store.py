"""Lưu frame, trạng thái review, correction log dưới dạng file JSON trong workspace."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from src.models.schemas import AuditItem, FrameRecord, VideoRecord
from src.models.schemas3d import Frame3DRecord

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
WORKSPACE_PREFIX = "@workspace/"


class WorkspaceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.corrections_file = self.root / "corrections.jsonl"
        self.exports_dir = self.root / "exports"
        self.videos_dir = self.root / "videos"
        # QC: mẫu audit ngẫu nhiên + nhật ký QC chỉ ghi thêm (ack warning, kết quả audit)
        self.qc_dir = self.root / "qc"
        self.audit_file = self.qc_dir / "audit.json"
        self.qc_log_file = self.qc_dir / "qc_log.jsonl"
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[int, int], FrameRecord]] = {}

    @staticmethod
    def check_id(frame_id: str) -> str:
        if not _SAFE_ID.match(frame_id):
            raise ValueError(f"frame_id không hợp lệ: {frame_id!r}")
        return frame_id

    def _write_json(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows: không thay được file khi có request khác đang đọc nó (UI hỏi tiến độ) -> đợi rồi thử lại
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))

    # ---- frame ----

    def frame_path(self, frame_id: str) -> Path:
        return self.frames_dir / f"{self.check_id(frame_id)}.json"

    def frame_ids(self) -> list[str]:
        if not self.frames_dir.is_dir():
            return []
        return sorted(p.stem for p in self.frames_dir.glob("*.json"))

    @staticmethod
    def _file_key(path: Path) -> tuple[int, int]:
        st = path.stat()
        return st.st_mtime_ns, st.st_size

    def load_frame(self, frame_id: str, copy: bool = True) -> FrameRecord | None:
        """copy=False trả về bản trong cache: chỉ dùng để đọc, không được sửa."""
        path = self.frame_path(frame_id)
        if not path.exists():
            return None
        key = self._file_key(path)
        cached = self._cache.get(frame_id)
        if cached and cached[0] == key:
            record = cached[1]
        else:
            record = FrameRecord.model_validate_json(path.read_text(encoding="utf-8"))
            self._cache[frame_id] = (key, record)
        return record.model_copy(deep=True) if copy else record

    def save_frame(self, record: FrameRecord) -> None:
        path = self.frame_path(record.frame_id)
        with self._lock:
            self._write_json(path, record.model_dump_json())
            # mtime trên Windows chỉ nhảy theo tick hệ thống, nên cập nhật cache ngay thay vì dựa vào mtime
            self._cache[record.frame_id] = (self._file_key(path), record.model_copy(deep=True))

    def list_frames(self) -> list[FrameRecord]:
        """Danh sách frame chỉ để đọc (không copy)."""
        return [f for f in (self.load_frame(fid, copy=False) for fid in self.frame_ids()) if f is not None]

    # ---- dữ liệu phụ theo frame: lidar chiếu xuống ảnh, GT 2D ----

    def save_aux(self, kind: str, frame_id: str, data) -> None:
        self._write_json(self.root / kind / f"{self.check_id(frame_id)}.json", data)

    def load_aux(self, kind: str, frame_id: str):
        path = self.root / kind / f"{self.check_id(frame_id)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ---- video tải lên ----
    # videos/<id>.json là record; videos/<id>/ chứa file gốc và frame đã cắt

    def video_dir(self, video_id: str) -> Path:
        return self.videos_dir / self.check_id(video_id)

    def save_video(self, video: VideoRecord) -> None:
        with self._lock:
            self._write_json(self.videos_dir / f"{self.check_id(video.video_id)}.json", video.model_dump_json())

    def load_video(self, video_id: str) -> VideoRecord | None:
        try:
            path = self.videos_dir / f"{self.check_id(video_id)}.json"
        except ValueError:
            return None
        return VideoRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_videos(self) -> list[VideoRecord]:
        if not self.videos_dir.is_dir():
            return []
        return [
            VideoRecord.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(self.videos_dir.glob("*.json"))
        ]

    def resolve(self, path: str) -> Path:
        """Đường dẫn ảnh của frame video tải lên được lưu dạng '@workspace/...' (tương đối so với workspace)."""
        return self.root / path.removeprefix(WORKSPACE_PREFIX)

    # ---- phần 3D: frames3d/<mô hình>/<frame_id>.json, point cloud rút gọn cho UI ở lidar3d/<frame_id>.bin ----

    def frame3d_path(self, model: str, frame_id: str) -> Path:
        return self.root / "frames3d" / self.check_id(model) / f"{self.check_id(frame_id)}.json"

    def models3d(self) -> list[str]:
        d = self.root / "frames3d"
        return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.json"))) if d.is_dir() else []

    def frame3d_ids(self, model: str) -> list[str]:
        d = self.root / "frames3d" / self.check_id(model)
        return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

    def load_frame3d(self, model: str, frame_id: str) -> Frame3DRecord | None:
        try:
            path = self.frame3d_path(model, frame_id)
        except ValueError:
            return None
        return Frame3DRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def save_frame3d(self, record: Frame3DRecord) -> None:
        with self._lock:
            self._write_json(self.frame3d_path(record.model, record.frame_id), record.model_dump_json())

    def list_frames3d(self, model: str) -> list[Frame3DRecord]:
        return [f for f in (self.load_frame3d(model, fid) for fid in self.frame3d_ids(model)) if f is not None]

    def points_path(self, frame_id: str) -> Path:
        return self.root / "lidar3d" / f"{self.check_id(frame_id)}.bin"

    # ---- correction log ----

    def append_corrections(self, entries: list[dict]) -> None:
        if not entries:
            return
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.corrections_file, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- sự kiện cấp frame (approve / reject / reopen / undo / redo): lịch sử duyệt và năng suất (FR-16, FR-27) ----

    @property
    def events_file(self) -> Path:
        return self.root / "events.jsonl"

    def append_event(self, event: dict) -> None:
        from datetime import UTC, datetime

        event = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def events(self) -> list[dict]:
        if not self.events_file.exists():
            return []
        with open(self.events_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- lịch sử undo / redo của từng frame (FR-09) ----

    def history_path(self, kind: str, frame_id: str) -> Path:
        return self.root / "history" / self.check_id(kind) / f"{self.check_id(frame_id)}.json"

    def load_history(self, kind: str, frame_id: str) -> dict:
        path = self.history_path(kind, frame_id)
        if not path.exists():
            return {"undo": [], "redo": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_history(self, kind: str, frame_id: str, history: dict) -> None:
        self._write_json(self.history_path(kind, frame_id), history)

    def corrections(self, frame_id: str | None = None) -> list[dict]:
        return self._read_jsonl(self.corrections_file, frame_id)

    @staticmethod
    def _read_jsonl(path: Path, frame_id: str | None = None) -> list[dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        return [e for e in entries if frame_id is None or e.get("frame_id") == frame_id]

    # ---- QC ----

    def load_audit(self) -> list[AuditItem]:
        if not self.audit_file.exists():
            return []
        data = json.loads(self.audit_file.read_text(encoding="utf-8"))
        return [AuditItem.model_validate(x) for x in data.get("items", [])]

    def save_audit(self, items: list[AuditItem]) -> None:
        with self._lock:
            self._write_json(self.audit_file, {"items": [i.model_dump() for i in items]})

    def append_qc_events(self, entries: list[dict]) -> None:
        if not entries:
            return
        with self._lock:
            self.qc_dir.mkdir(parents=True, exist_ok=True)
            with open(self.qc_log_file, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def qc_events(self, frame_id: str | None = None) -> list[dict]:
        return self._read_jsonl(self.qc_log_file, frame_id)
