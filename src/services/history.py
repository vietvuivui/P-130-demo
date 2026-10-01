"""Undo / redo thao tác duyệt trên một frame (FR-09), dùng chung cho frame 2D và frame 3D.

Trước mỗi thao tác làm đổi nhãn (KEEP / DELETE / đổi lớp / sửa box / thêm box / duyệt theo lô), bản frame ngay trước đó
được đẩy vào ngăn undo của frame (tối đa MAX_STEPS bước, lưu ở workspace/history/<kind>/<frame>.json). Undo lấy lại bản
đó và đẩy bản hiện tại sang ngăn redo; một thao tác mới xoá ngăn redo. Approve / reopen / reject không vào lịch sử:
frame đã approve phải mở lại trước khi undo.
"""

from __future__ import annotations

from typing import Literal

from src.services.store import WorkspaceStore

MAX_STEPS = 50


class HistoryError(ValueError):
    pass


def kind_2d() -> str:
    return "2d"


def kind_3d(model: str) -> str:
    return f"3d-{model}"


def record(store: WorkspaceStore, kind: str, frame_id: str, before_json: str) -> None:
    """Gọi sau khi một thao tác thành công, với bản JSON của frame trước thao tác."""
    h = store.load_history(kind, frame_id)
    h["undo"] = (h["undo"] + [before_json])[-MAX_STEPS:]
    h["redo"] = []
    store.save_history(kind, frame_id, h)


def step(store: WorkspaceStore, kind: str, frame_id: str, current_json: str, direction: Literal["undo", "redo"]) -> str:
    """Trả về bản JSON của frame sau khi undo / redo, và cập nhật hai ngăn."""
    h = store.load_history(kind, frame_id)
    other = "redo" if direction == "undo" else "undo"
    if not h[direction]:
        raise HistoryError("Không còn thao tác để hoàn tác" if direction == "undo" else "Không còn thao tác để làm lại")
    target = h[direction].pop()
    h[other] = (h[other] + [current_json])[-MAX_STEPS:]
    store.save_history(kind, frame_id, h)
    return target


def counts(store: WorkspaceStore, kind: str, frame_id: str) -> dict:
    h = store.load_history(kind, frame_id)
    return {"undo": len(h["undo"]), "redo": len(h["redo"])}


def changed_objects(before, after) -> list:
    """Object có trạng thái duyệt / box khác nhau giữa hai bản frame (để ghi log UNDO / REDO)."""
    old = {o.object_id: o for o in before.objects}
    out = []
    for o in after.objects:
        prev = old.pop(o.object_id, None)
        if prev is None or prev.review != o.review or getattr(prev, "box", None) != getattr(o, "box", None):
            out.append(o)
    return out + list(old.values())  # object biến mất (box người vẽ bị hoàn tác)
