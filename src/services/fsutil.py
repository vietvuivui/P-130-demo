"""Thao tác file dùng chung."""

from __future__ import annotations

import os
import time
from pathlib import Path


def replace_file(tmp: str | Path, path: str | Path, attempts: int = 40, delay_s: float = 0.05) -> None:
    """os.replace có thử lại. Windows không cho thay file đang được mở ở chỗ khác (request khác đang đọc để báo tiến
    độ, OneDrive đang đồng bộ, phần mềm diệt virus đang quét) và ném PermissionError [WinError 5]; file thường được nhả
    sau vài chục mili giây. Hết số lần thử (mặc định ~2 s) mới ném lỗi."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)
