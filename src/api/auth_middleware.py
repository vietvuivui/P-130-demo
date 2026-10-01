"""Middleware: bắt đăng nhập (AUTH_REQUIRED=1) và chặn ghi đè frame người khác đang mở (khoá frame, collab.json)."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from src.config import get_settings

PUBLIC_PREFIXES = ("/api/v1/auth/", "/api/v1/invites/", "/health", "/api/v1/projects/env")
WRITE_OPS = {"actions", "approve", "approve-low-risk", "reject", "undo", "redo", "reopen", "propagate", "sweeps", "qc"}


def frame_of_write(path: str) -> tuple[str | None, str | None] | None:
    """(project_id, frame_id) nếu là POST sửa frame: /api/v1/frames/{fid}/... (2D), /frames/{model}/{fid}/... (3D),
    có thể nằm sau /p/{pid}. None nếu không phải."""
    pid = None
    if path.startswith("/p/"):
        parts = path.split("/", 3)
        if len(parts) < 4:
            return None
        pid, path = parts[2], "/" + parts[3]
    if not path.startswith("/api/v1/frames/"):
        return None
    segs = [s for s in path[len("/api/v1/frames/") :].split("/") if s]
    if len(segs) >= 2 and segs[1] in WRITE_OPS:
        return pid, segs[0]
    if len(segs) >= 3 and segs[2] in WRITE_OPS:
        return pid, segs[1]
    return None


def _store_root(pid: str | None):
    if pid:
        from src.services.projects import get_manager

        return get_manager().store(pid).root
    return get_settings().workspace_dir


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    is_api = path.startswith("/api/") or (path.startswith("/p/") and "/api/v1/" in path)
    if not is_api:
        return await call_next(request)
    s = get_settings()
    from src.api.auth_routes import current_user

    user = current_user(request)
    if s.auth_required and user is None and not path.startswith(PUBLIC_PREFIXES):
        return JSONResponse(status_code=401, content={"detail": {"code": "LOGIN_REQUIRED", "message": "Cần đăng nhập"}})
    if user is not None and request.method == "POST":
        hit = frame_of_write(path)
        if hit:
            from src.services.users import AuthError, Collab

            pid, fid = hit
            try:
                Collab(_store_root(pid)).acquire(fid, user.id)  # giữ / gia hạn khoá cho người đang sửa
            except AuthError as e:
                return JSONResponse(status_code=e.status, content={"detail": {"code": e.code, "message": str(e)}})
            except Exception:  # dự án không tồn tại...: để route trả lỗi đúng
                pass
    return await call_next(request)
