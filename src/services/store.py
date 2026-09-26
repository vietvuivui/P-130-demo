"""Lưu frame, trạng thái review, correction log dưới dạng file JSON trong workspace."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from src.models.schemas import FrameRecord

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class WorkspaceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.corrections_file = self.root / "corrections.jsonl"
        self.exports_dir = self.root / "exports"
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
        os.replace(tmp, path)

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

    # ---- correction log ----

    def append_corrections(self, entries: list[dict]) -> None:
        if not entries:
            return
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.corrections_file, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def corrections(self, frame_id: str | None = None) -> list[dict]:
        if not self.corrections_file.exists():
            return []
        with open(self.corrections_file, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        return [e for e in entries if frame_id is None or e["frame_id"] == frame_id]
