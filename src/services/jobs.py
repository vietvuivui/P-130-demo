"""Việc chạy nền cho các nút trên UI (áp dụng lại cài đặt, đánh giá trước / sau): một việc mỗi loại mỗi workspace.

Trạng thái chỉ giữ trong bộ nhớ (mất khi tắt server); kết quả lâu dài do chính việc đó ghi vào workspace.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def status(key: str) -> dict:
    with _LOCK:
        return dict(_JOBS.get(key) or {"state": "idle"})


def start(key: str, fn: Callable[[Callable[[float, str], None]], dict]) -> dict:
    """Chạy fn(progress) ở thread nền. progress(0..1, lời nhắn). Trả về trạng thái; đang chạy thì không chạy thêm."""
    with _LOCK:
        cur = _JOBS.get(key)
        if cur and cur["state"] == "running":
            return dict(cur)
        _JOBS[key] = {"state": "running", "progress": 0.0, "message": "Đang bắt đầu…", "started": time.time()}

    def progress(p: float, msg: str) -> None:
        with _LOCK:
            _JOBS[key].update(progress=round(max(0.0, min(1.0, p)), 3), message=msg)

    def run() -> None:
        try:
            result = fn(progress)
            with _LOCK:
                _JOBS[key].update(state="done", progress=1.0, message="Xong", result=result, finished=time.time())
        except Exception as e:  # báo lỗi lên UI, không làm sập server
            log.exception("Việc nền %s lỗi", key)
            with _LOCK:
                _JOBS[key].update(state="error", message=f"{type(e).__name__}: {e}", finished=time.time())

    threading.Thread(target=run, daemon=True, name=f"job:{key}").start()
    return status(key)


def wait(key: str, timeout: float = 60) -> dict:
    """Chờ việc xong (dùng trong test)."""
    end = time.time() + timeout
    while time.time() < end and status(key).get("state") == "running":
        time.sleep(0.05)
    return status(key)
