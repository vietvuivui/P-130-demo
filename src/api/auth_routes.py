"""Đăng nhập, tài khoản, lời mời, thành viên dự án, chia việc, khoá frame (src/services/users.py).

- /api/v1/auth/*: register (người đầu tiên = admin), login, logout, me, users (admin)
- /api/v1/invites/{token}: xem / nhận lời mời (tạo tài khoản luôn nếu email chưa có)
- /api/v1/projects/{pid}/members|invites|assignments|split: owner quản lý; thành viên xem
- /api/v1/frames/{id}/lock, /api/v1/presence: nằm ở routes.py (dùng chung cho workspace mặc định và /p/{pid})
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.config import get_settings
from src.services.projects import ProjectError, get_manager
from src.services.users import COOKIE, ROLES, AuthError, Collab, User, get_user_store, send_invite_email

auth_router = APIRouter()


def _err(e: AuthError) -> HTTPException:
    return HTTPException(status_code=e.status, detail={"code": e.code, "message": str(e)})


def current_user(request: Request) -> User | None:
    return get_user_store().user_of_session(request.cookies.get(COOKIE))


def require_user(request: Request) -> User:
    u = current_user(request)
    if u is None:
        raise HTTPException(status_code=401, detail={"code": "LOGIN_REQUIRED", "message": "Cần đăng nhập"})
    return u


def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=30 * 24 * 3600, path="/")


def member_role(project, user: User | None) -> str | None:
    if user is None:
        return None
    if user.admin:
        return "owner"
    for m in project.members:
        if m.get("user_id") == user.id:
            return m.get("role")
    return None


def require_member(pid: str, user: User, roles: tuple[str, ...] = ROLES):
    try:
        p = get_manager().get(pid)
    except ProjectError as e:
        raise HTTPException(status_code=e.status, detail={"code": e.code, "message": str(e)}) from e
    role = member_role(p, user)
    if role is None and p.members:
        raise HTTPException(status_code=403, detail={"code": "NOT_MEMBER", "message": "Bạn không ở trong dự án này"})
    if role not in roles and p.members:
        raise HTTPException(status_code=403, detail={"code": "ROLE_REQUIRED", "message": f"Cần vai trò {' / '.join(roles)}"})
    return p


# ---------------------------------------------------------------- tài khoản


class RegisterIn(BaseModel):
    email: str
    name: str = ""
    password: str = Field(min_length=6)
    invite: str | None = None


class LoginIn(BaseModel):
    email: str
    password: str


@auth_router.post("/auth/register")
def register(body: RegisterIn, response: Response):
    """Người đầu tiên thành admin. Sau đó cần link mời (hoặc AUTH_OPEN_SIGNUP=1)."""
    us = get_user_store()
    s = get_settings()
    inv = None
    try:
        if us.users() and not s.auth_open_signup:
            if not body.invite:
                raise AuthError("INVITE_REQUIRED", "Cần link mời để tạo tài khoản", 403)
            inv = us.get_invite(body.invite)
            if inv.email != body.email.strip().lower():
                raise AuthError("INVITE_EMAIL", f"Link mời này dành cho {inv.email}", 403)
        u = us.create(body.email, body.name, body.password)
        if inv is not None:
            _accept(inv, u)
    except AuthError as e:
        raise _err(e) from e
    _set_cookie(response, us.new_session(u.id))
    return u.public()


@auth_router.post("/auth/login")
def login(body: LoginIn, response: Response):
    try:
        u, token = get_user_store().login(body.email, body.password)
    except AuthError as e:
        raise _err(e) from e
    _set_cookie(response, token)
    return u.public()


@auth_router.post("/auth/logout")
def logout(request: Request, response: Response):
    get_user_store().logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@auth_router.get("/auth/me")
def me(request: Request):
    u = current_user(request)
    s = get_settings()
    return {"user": u.public() if u else None, "auth_required": s.auth_required,
            "has_users": bool(get_user_store().users()), "open_signup": s.auth_open_signup}  # fmt: skip


@auth_router.get("/auth/users")
def list_users(user: User = Depends(require_user)):
    return [u.public() for u in get_user_store().users()]


# ---------------------------------------------------------------- lời mời


class InviteIn(BaseModel):
    email: str
    role: str = "annotator"


def _invite_link(request: Request, token: str) -> str:
    base = get_settings().public_url.rstrip("/") or str(request.base_url).rstrip("/")
    return f"{base}/ui/login.html?invite={token}"


def _accept(inv, user: User) -> None:
    mgr = get_manager()
    p = mgr.get(inv.project_id)
    if not any(m.get("user_id") == user.id for m in p.members):
        p.members.append({"user_id": user.id, "role": inv.role})
        mgr.save(p)
    get_user_store().mark_accepted(inv.token, user.id)


@auth_router.post("/projects/{pid}/invites")
def create_invite(pid: str, body: InviteIn, request: Request, user: User = Depends(require_user)):
    """Tạo lời mời; trả link để gửi tay (và gửi mail nếu có SMTP). Người được mời mở link, đăng ký / đăng nhập là vào."""
    p = require_member(pid, user, ("owner",))
    us = get_user_store()
    try:
        inv = us.invite(pid, body.email, body.role, user.id)
    except AuthError as e:
        raise _err(e) from e
    link = _invite_link(request, inv.token)
    existing = us.by_email(inv.email)
    emailed = False
    try:
        emailed = send_invite_email(inv, link)
    except Exception:  # SMTP lỗi: vẫn trả link
        emailed = False
    return {"token": inv.token, "link": link, "email": inv.email, "role": inv.role, "emailed": emailed,
            "project": p.name, "has_account": existing is not None}  # fmt: skip


@auth_router.get("/projects/{pid}/invites")
def list_invites(pid: str, request: Request, user: User = Depends(require_user)):
    require_member(pid, user, ("owner",))
    return [{**i.model_dump(), "link": _invite_link(request, i.token)} for i in get_user_store().invites_of(pid)]


@auth_router.get("/invites/{token}")
def invite_info(token: str):
    try:
        inv = get_user_store().get_invite(token)
        p = get_manager().get(inv.project_id)
    except AuthError as e:
        raise _err(e) from e
    except ProjectError as e:
        raise HTTPException(status_code=404, detail={"code": e.code, "message": str(e)}) from e
    return {"email": inv.email, "role": inv.role, "project": p.name, "project_id": p.id,
            "has_account": get_user_store().by_email(inv.email) is not None}  # fmt: skip


@auth_router.post("/invites/{token}/accept")
def accept_invite(token: str, request: Request):
    """Người đã đăng nhập nhận lời mời (email tài khoản phải trùng email được mời)."""
    user = require_user(request)
    try:
        inv = get_user_store().get_invite(token)
        if inv.email != user.email:
            raise AuthError("INVITE_EMAIL", f"Link mời này dành cho {inv.email}", 403)
        _accept(inv, user)
    except AuthError as e:
        raise _err(e) from e
    return {"ok": True, "project_id": inv.project_id, "role": inv.role}


# ---------------------------------------------------------------- thành viên


class MemberIn(BaseModel):
    user_id: str
    role: str = "annotator"


def _members_view(p) -> list[dict]:
    us = get_user_store()
    out = []
    for m in p.members:
        u = us.get(m.get("user_id", ""))
        out.append({**m, "name": u.name if u else "?", "email": u.email if u else ""})
    return out


@auth_router.get("/projects/{pid}/members")
def list_members(pid: str, request: Request):
    user = current_user(request)
    p = require_member(pid, user, ROLES) if user else get_manager().get(pid)
    return {"members": _members_view(p), "my_role": member_role(p, user)}


@auth_router.post("/projects/{pid}/members")
def add_member(pid: str, body: MemberIn, user: User = Depends(require_user)):
    """Thêm thẳng người đã có tài khoản (không qua link mời) hoặc đổi vai trò."""
    p = require_member(pid, user, ("owner",))
    if body.role not in ROLES:
        raise HTTPException(status_code=422, detail={"code": "ROLE_INVALID", "message": "Vai trò không hợp lệ"})
    if get_user_store().get(body.user_id) is None:
        raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": "Không có người dùng này"})
    for m in p.members:
        if m.get("user_id") == body.user_id:
            m["role"] = body.role
            break
    else:
        p.members.append({"user_id": body.user_id, "role": body.role})
    get_manager().save(p)
    return {"members": _members_view(p)}


@auth_router.delete("/projects/{pid}/members/{user_id}")
def remove_member(pid: str, user_id: str, user: User = Depends(require_user)):
    p = require_member(pid, user, ("owner",))
    owners = [m for m in p.members if m.get("role") == "owner"]
    if any(m.get("user_id") == user_id and m.get("role") == "owner" for m in p.members) and len(owners) <= 1:
        raise HTTPException(status_code=409, detail={"code": "LAST_OWNER", "message": "Dự án cần ít nhất một owner"})
    p.members = [m for m in p.members if m.get("user_id") != user_id]
    get_manager().save(p)
    Collab(get_manager().store(pid).root).assign(
        [f for f, u in Collab(get_manager().store(pid).root).assignments().items() if u == user_id], None
    )
    return {"members": _members_view(p)}


# ---------------------------------------------------------------- chia việc


class AssignIn(BaseModel):
    frame_ids: list[str]
    user_id: str | None = None  # None = bỏ giao


class SplitIn(BaseModel):
    user_ids: list[str]
    scope: str = "pending"  # pending: frame chưa duyệt | all
    keep_existing: bool = True


def _frame_ids(pid: str, scope: str) -> list[str]:
    store = get_manager().store(pid)
    frames = sorted(store.list_frames(), key=lambda f: (f.scene, f.index))
    if scope == "pending":
        frames = [f for f in frames if f.status != "approved"]
    return [f.frame_id for f in frames]


def _summary(pid: str, assignments: dict[str, str]) -> dict:
    store = get_manager().store(pid)
    status = {f.frame_id: f.status for f in store.list_frames()}
    per: dict[str, dict] = {}
    for fid, uid in assignments.items():
        d = per.setdefault(uid, {"assigned": 0, "approved": 0})
        d["assigned"] += 1
        if status.get(fid) == "approved":
            d["approved"] += 1
    us = get_user_store()
    return {uid: {**d, "name": (us.get(uid).name if us.get(uid) else "?")} for uid, d in per.items()}


@auth_router.get("/projects/{pid}/assignments")
def get_assignments(pid: str, request: Request):
    user = current_user(request)
    if user:
        require_member(pid, user, ROLES)
    a = Collab(get_manager().store(pid).root).assignments()
    return {"assignments": a, "summary": _summary(pid, a), "unassigned": len(set(_frame_ids(pid, "all")) - set(a))}


@auth_router.post("/projects/{pid}/assignments")
def assign_frames(pid: str, body: AssignIn, user: User = Depends(require_user)):
    require_member(pid, user, ("owner", "reviewer"))
    a = Collab(get_manager().store(pid).root).assign(body.frame_ids, body.user_id)
    return {"assignments": a, "summary": _summary(pid, a)}


@auth_router.post("/projects/{pid}/split")
def split_frames(pid: str, body: SplitIn, user: User = Depends(require_user)):
    """Chia đều frame (chưa duyệt, theo thứ tự scene / thời gian) cho các thành viên được chọn."""
    p = require_member(pid, user, ("owner", "reviewer"))
    ids = {m.get("user_id") for m in p.members} | ({user.id} if user.admin else set())
    bad = [u for u in body.user_ids if u not in ids]
    if bad:
        raise HTTPException(status_code=422, detail={"code": "NOT_MEMBER", "message": f"Không phải thành viên: {', '.join(bad)}"})
    try:
        a = Collab(get_manager().store(pid).root).split(_frame_ids(pid, body.scope), body.user_ids, body.keep_existing)
    except AuthError as e:
        raise _err(e) from e
    return {"assignments": a, "summary": _summary(pid, a)}
