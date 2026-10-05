"""Người dùng, phiên đăng nhập, lời mời vào dự án, thành viên / vai trò, chia việc theo frame và khoá frame.

Mô hình làm việc nhiều người (tham khảo CVAT + cách mời của Google Docs), đủ cho nhóm nhỏ, không cần CSDL:
- `data/users.json`: tài khoản (email, tên, mật khẩu băm PBKDF2), phiên (cookie `al_session`), lời mời.
  Người đăng ký đầu tiên là admin. Về sau chỉ vào được qua link mời (hoặc admin tạo), trừ khi `AUTH_OPEN_SIGNUP=1`.
- Thành viên dự án nằm trong `project.json` (`members: [{user_id, role}]`, role = owner | reviewer | annotator):
  owner mời người, chia việc; reviewer duyệt / trả lại frame; annotator gán nhãn frame được giao.
- Chia việc + khoá frame nằm trong workspace của dự án (`collab.json`): `assignments[frame_id] = user_id`;
  `locks[frame_id] = {user_id, at}` — ai mở frame thì giữ khoá (gia hạn mỗi 30 s, hết hạn sau LOCK_TTL_S), người khác
  thấy "đang mở" và không ghi đè (API trả 409 FRAME_LOCKED). Không có WebSocket: UI hỏi lại /presence mỗi 15 s.

Bật bắt buộc đăng nhập bằng `AUTH_REQUIRED=1` trong .env (mặc định tắt để giữ cách chạy một người và test cũ).
Gửi mail: chưa có SMTP, server trả link mời để người tạo gửi tay (chỗ cắm SMTP: `send_invite_email`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.services.fsutil import replace_file

Role = Literal["owner", "reviewer", "annotator"]
ROLES: tuple[str, ...] = ("owner", "reviewer", "annotator")
SESSION_TTL_S = 30 * 24 * 3600
INVITE_TTL_S = 7 * 24 * 3600
LOCK_TTL_S = 90
COOKIE = "al_session"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class User(BaseModel):
    id: str
    email: str
    name: str
    admin: bool = False
    created_at: float
    password_hash: str = ""  # "pbkdf2$<iter>$<salt>$<hex>"

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name, "admin": self.admin}


class Invite(BaseModel):
    token: str
    project_id: str
    email: str
    role: str
    created_by: str
    created_at: float
    expires_at: float
    accepted_by: str | None = None


class AuthError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def hash_password(pw: str, iterations: int = 120_000) -> str:
    salt = secrets.token_hex(8)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return f"pbkdf2${iterations}${salt}${h}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), int(it)).hex()
        return hmac.compare_digest(got, h)
    except ValueError:
        return False


class UserStore:
    """users.json: {"users": {...}, "sessions": {token: {user_id, expires_at}}, "invites": {token: {...}}}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict | None = None

    # ---- lưu / đọc
    def _load(self) -> dict:
        if self._data is None:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._data = {"users": {}, "sessions": {}, "invites": {}}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, ensure_ascii=False), encoding="utf-8")
        replace_file(tmp, self.path)

    # ---- người dùng
    def users(self) -> list[User]:
        with self._lock:
            return [User(**u) for u in self._load()["users"].values()]

    def get(self, user_id: str) -> User | None:
        with self._lock:
            u = self._load()["users"].get(user_id)
            return User(**u) if u else None

    def by_email(self, email: str) -> User | None:
        email = email.strip().lower()
        with self._lock:
            for u in self._load()["users"].values():
                if u["email"] == email:
                    return User(**u)
        return None

    def create(self, email: str, name: str, password: str, admin: bool | None = None) -> User:
        email = email.strip().lower()
        if not _EMAIL.match(email):
            raise AuthError("EMAIL_INVALID", "Email không hợp lệ")
        if len(password) < 6:
            raise AuthError("PASSWORD_SHORT", "Mật khẩu cần ít nhất 6 ký tự")
        with self._lock:
            d = self._load()
            if self.by_email(email):
                raise AuthError("EMAIL_TAKEN", "Email này đã có tài khoản", 409)
            first = not d["users"]
            u = User(id=f"u-{secrets.token_hex(4)}", email=email, name=(name or email.split("@")[0]).strip()[:60],
                     admin=first if admin is None else admin, created_at=time.time(),
                     password_hash=hash_password(password))  # fmt: skip
            d["users"][u.id] = u.model_dump()
            self._save()
            return u

    def set_password(self, user_id: str, password: str) -> None:
        if len(password) < 6:
            raise AuthError("PASSWORD_SHORT", "Mật khẩu cần ít nhất 6 ký tự")
        with self._lock:
            d = self._load()
            d["users"][user_id]["password_hash"] = hash_password(password)
            self._save()

    # ---- phiên
    def login(self, email: str, password: str) -> tuple[User, str]:
        u = self.by_email(email)
        if u is None or not check_password(password, u.password_hash):
            raise AuthError("LOGIN_FAILED", "Sai email hoặc mật khẩu", 401)
        return u, self.new_session(u.id)

    def new_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            d = self._load()
            now = time.time()
            d["sessions"] = {t: s for t, s in d["sessions"].items() if s["expires_at"] > now}
            d["sessions"][token] = {"user_id": user_id, "expires_at": now + SESSION_TTL_S}
            self._save()
        return token

    def user_of_session(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._lock:
            s = self._load()["sessions"].get(token)
            if not s or s["expires_at"] < time.time():
                return None
            return self.get(s["user_id"])

    def logout(self, token: str | None) -> None:
        with self._lock:
            if token and token in self._load()["sessions"]:
                del self._data["sessions"][token]
                self._save()

    # ---- lời mời
    def invite(self, project_id: str, email: str, role: str, created_by: str) -> Invite:
        email = email.strip().lower()
        if not _EMAIL.match(email):
            raise AuthError("EMAIL_INVALID", "Email không hợp lệ")
        if role not in ROLES:
            raise AuthError("ROLE_INVALID", f"Vai trò phải là một trong {', '.join(ROLES)}")
        inv = Invite(token=secrets.token_urlsafe(18), project_id=project_id, email=email, role=role,
                     created_by=created_by, created_at=time.time(), expires_at=time.time() + INVITE_TTL_S)  # fmt: skip
        with self._lock:
            d = self._load()
            d["invites"][inv.token] = inv.model_dump()
            self._save()
        return inv

    def get_invite(self, token: str) -> Invite:
        with self._lock:
            raw = self._load()["invites"].get(token)
        if raw is None:
            raise AuthError("INVITE_NOT_FOUND", "Link mời không tồn tại", 404)
        inv = Invite(**raw)
        if inv.accepted_by:
            raise AuthError("INVITE_USED", "Link mời này đã được dùng", 410)
        if inv.expires_at < time.time():
            raise AuthError("INVITE_EXPIRED", "Link mời đã hết hạn (7 ngày)", 410)
        return inv

    def invites_of(self, project_id: str) -> list[Invite]:
        with self._lock:
            return [Invite(**i) for i in self._load()["invites"].values() if i["project_id"] == project_id]

    def mark_accepted(self, token: str, user_id: str) -> None:
        with self._lock:
            d = self._load()
            d["invites"][token]["accepted_by"] = user_id
            self._save()


def send_invite_email(inv: Invite, link: str) -> bool:  # pragma: no cover - chưa có SMTP
    """Chỗ cắm SMTP (SMTP_HOST / SMTP_USER / SMTP_PASSWORD / SMTP_FROM trong .env). Chưa cấu hình: trả False, UI
    hiện link để gửi tay."""
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Mời tham gia dự án gán nhãn AutoLabel 3D"
    msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "autolabel@localhost"))
    msg["To"] = inv.email
    msg.set_content(f"Bạn được mời vào dự án với vai trò {inv.role}. Mở link để nhận việc: {link}")
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as s:
        s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        s.send_message(msg)
    return True


# ---------------------------------------------------------------------------
# Chia việc + khoá frame trong workspace (collab.json)
# ---------------------------------------------------------------------------


class Collab:
    """assignments / locks của một workspace. Khoá có hạn LOCK_TTL_S; gia hạn bằng cách gọi lại acquire."""

    def __init__(self, root: str | Path):
        self.path = Path(root) / "collab.json"
        self._lock = threading.RLock()

    def _load(self) -> dict:
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            d = {}
        d.setdefault("assignments", {})
        d.setdefault("locks", {})
        return d

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        replace_file(tmp, self.path)

    def assignments(self) -> dict[str, str]:
        with self._lock:
            return dict(self._load()["assignments"])

    def assign(self, frame_ids: list[str], user_id: str | None) -> dict[str, str]:
        """user_id=None: bỏ giao."""
        with self._lock:
            d = self._load()
            for fid in frame_ids:
                if user_id:
                    d["assignments"][fid] = user_id
                else:
                    d["assignments"].pop(fid, None)
            self._save(d)
            return dict(d["assignments"])

    def split(self, frame_ids: list[str], user_ids: list[str], keep_existing: bool = True) -> dict[str, str]:
        """Chia đều frame (theo thứ tự) cho các user, xoay vòng; keep_existing: frame đã giao thì giữ nguyên."""
        if not user_ids:
            raise AuthError("NO_USERS", "Chọn ít nhất một thành viên để chia việc")
        with self._lock:
            d = self._load()
            k = 0
            for fid in frame_ids:
                if keep_existing and fid in d["assignments"]:
                    continue
                d["assignments"][fid] = user_ids[k % len(user_ids)]
                k += 1
            self._save(d)
            return dict(d["assignments"])

    def locks(self) -> dict[str, dict]:
        now = time.time()
        with self._lock:
            d = self._load()
            live = {f: v for f, v in d["locks"].items() if now - v["at"] < LOCK_TTL_S}
            if len(live) != len(d["locks"]):
                d["locks"] = live
                self._save(d)
            return dict(live)

    def holder(self, frame_id: str) -> str | None:
        return (self.locks().get(frame_id) or {}).get("user_id")

    def acquire(self, frame_id: str, user_id: str, force: bool = False) -> dict:
        """Giữ / gia hạn khoá. Người khác đang giữ (chưa hết hạn) -> AuthError FRAME_LOCKED, trừ khi force."""
        with self._lock:
            d = self._load()
            now = time.time()
            cur = d["locks"].get(frame_id)
            if cur and cur["user_id"] != user_id and now - cur["at"] < LOCK_TTL_S and not force:
                raise AuthError("FRAME_LOCKED", "Người khác đang mở frame này", 409)
            d["locks"][frame_id] = {"user_id": user_id, "at": now}
            # Mỗi người chỉ giữ một frame: thả các frame khác của cùng người
            for f in [f for f, v in d["locks"].items() if v["user_id"] == user_id and f != frame_id]:
                del d["locks"][f]
            self._save(d)
            return d["locks"][frame_id]

    def release(self, frame_id: str, user_id: str) -> None:
        with self._lock:
            d = self._load()
            cur = d["locks"].get(frame_id)
            if cur and cur["user_id"] == user_id:
                del d["locks"][frame_id]
                self._save(d)


_store: UserStore | None = None


def get_user_store() -> UserStore:
    global _store
    if _store is None:
        from src.config import get_settings

        _store = UserStore(Path(get_settings().users_file))
    return _store
