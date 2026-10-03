"""Nhiều người dùng: đăng ký / đăng nhập, mời theo link, thành viên, chia việc, khoá frame."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.services import projects, users
from src.services.projects import ProjectManager, ProjectOptions
from src.services.users import Collab, UserStore, check_password, hash_password
from tests.test_services.test_projects import demo_image


@pytest.fixture
def user_store(tmp_path, monkeypatch):
    us = UserStore(tmp_path / "users.json")
    monkeypatch.setattr(users, "_store", us)
    return us


@pytest.fixture
def manager(tmp_path, config, monkeypatch):
    cfg = config.model_copy(deep=True)
    cfg.detection.detectors = ["demo"]
    m = ProjectManager(tmp_path / "projects", lambda: cfg, mm3d_python=str(tmp_path / "no-mm3d"))
    monkeypatch.setattr(projects, "_manager", m)
    return m


@pytest.fixture
def project(manager, tmp_path):
    files = []
    for i in range(4):
        f = tmp_path / f"img{i}.jpg"
        demo_image(f)
        files.append(f)
    p = manager.create("Nhóm", files, options=ProjectOptions(sequential=False))
    return manager.run(p.id)


@pytest_asyncio.fixture
async def api(user_store, manager):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_password_hash():
    h = hash_password("secret1")
    assert check_password("secret1", h) and not check_password("secret2", h) and not check_password("x", "bad")


def test_collab_assign_split_and_locks(tmp_path):
    c = Collab(tmp_path / "ws")
    a = c.split(["f1", "f2", "f3", "f4", "f5"], ["u1", "u2"])
    assert a == {"f1": "u1", "f2": "u2", "f3": "u1", "f4": "u2", "f5": "u1"}
    c.assign(["f5"], None)
    assert "f5" not in c.assignments()
    c.split(["f1", "f5"], ["u3"])  # f1 đã giao thì giữ, f5 mới nhận
    assert c.assignments()["f1"] == "u1" and c.assignments()["f5"] == "u3"
    c.acquire("f1", "u1")
    with pytest.raises(users.AuthError, match="FRAME_LOCKED|đang mở"):
        c.acquire("f1", "u2")
    c.acquire("f2", "u1")  # mỗi người một frame: f1 được thả
    assert c.holder("f1") is None and c.holder("f2") == "u1"
    c.acquire("f2", "u2", force=True)
    assert c.holder("f2") == "u2"
    c.release("f2", "u2")
    assert c.locks() == {}


@pytest.mark.asyncio
async def test_register_login_invite_split_lock(api, user_store, project, manager):
    pid = project.id
    # Người đầu tiên = admin
    r = await api.post("/api/v1/auth/register", json={"email": "kien@x.vn", "name": "Kiên", "password": "secret1"})
    assert r.status_code == 200 and r.json()["admin"] is True
    me = (await api.get("/api/v1/auth/me")).json()
    assert me["user"]["email"] == "kien@x.vn"
    # Người thứ hai không có link mời -> bị chặn
    async with _client() as other:
        r = await other.post("/api/v1/auth/register", json={"email": "danh@x.vn", "password": "secret2"})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "INVITE_REQUIRED"
    # Mời Danh làm annotator -> link
    r = await api.post(f"/api/v1/projects/{pid}/invites", json={"email": "danh@x.vn", "role": "annotator"})
    assert r.status_code == 200, r.text
    inv = r.json()
    assert "/ui/login.html?invite=" in inv["link"] and inv["emailed"] is False
    info = (await api.get(f"/api/v1/invites/{inv['token']}")).json()
    assert info["project"] == "Nhóm" and info["role"] == "annotator" and info["has_account"] is False
    # Danh đăng ký qua link -> thành thành viên
    async with _client() as danh:
        r = await danh.post("/api/v1/auth/register",
                            json={"email": "danh@x.vn", "name": "Danh", "password": "secret2", "invite": inv["token"]})  # fmt: skip
        assert r.status_code == 200, r.text
        danh_id = r.json()["id"]
        m = (await danh.get(f"/api/v1/projects/{pid}/members")).json()
        assert m["my_role"] == "annotator" and [x["user_id"] for x in m["members"]] == [danh_id]
        # Link dùng rồi
        assert (await danh.get(f"/api/v1/invites/{inv['token']}")).status_code == 410
        # Annotator không được mời người khác
        r = await danh.post(f"/api/v1/projects/{pid}/invites", json={"email": "c@x.vn"})
        assert r.status_code == 403
        # Admin (Kiên) chia việc cho Danh và mình
        kien_id = me["user"]["id"]
        r = await api.post(f"/api/v1/projects/{pid}/split", json={"user_ids": [danh_id, kien_id]})
        assert r.status_code == 200, r.text
        a = r.json()["assignments"]
        assert len(a) == 4 and sorted(set(a.values())) == sorted([danh_id, kien_id])
        assert r.json()["summary"][danh_id]["assigned"] == 2
        # Khoá frame: Danh mở frame -> Kiên không ghi được (409), presence thấy Danh
        fid = next(iter(a))
        r = await danh.post(f"/p/{pid}/api/v1/frames/{fid}/lock")
        assert r.status_code == 200 and r.json()["locked"] is True
        pres = (await api.get(f"/p/{pid}/api/v1/presence")).json()
        assert pres["locks"][fid]["name"] == "Danh" and pres["assignments"] == a
        r = await api.post(f"/p/{pid}/api/v1/frames/{fid}/approve-low-risk", json={})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "FRAME_LOCKED"
        # Danh ghi được; thả khoá thì Kiên ghi được
        assert (await danh.post(f"/p/{pid}/api/v1/frames/{fid}/approve-low-risk", json={})).status_code == 200
        await danh.delete(f"/p/{pid}/api/v1/frames/{fid}/lock")
        assert (await api.post(f"/p/{pid}/api/v1/frames/{fid}/approve-low-risk", json={})).status_code == 200
        # Gỡ Danh khỏi dự án -> mất phần việc
        r = await api.delete(f"/api/v1/projects/{pid}/members/{danh_id}")
        assert r.status_code == 200 and r.json()["members"] == []
        assert danh_id not in (await api.get(f"/api/v1/projects/{pid}/assignments")).json()["assignments"].values()
    # Đăng xuất
    await api.post("/api/v1/auth/logout")
    assert (await api.get("/api/v1/auth/me")).json()["user"] is None
    assert (await api.post("/api/v1/auth/login", json={"email": "kien@x.vn", "password": "wrong"})).status_code == 401
    assert (await api.post("/api/v1/auth/login", json={"email": "kien@x.vn", "password": "secret1"})).status_code == 200


@pytest.mark.asyncio
async def test_auth_required_blocks_anonymous(api, user_store, monkeypatch):
    from src.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "auth_required", True)
    assert (await api.get("/api/v1/projects")).status_code == 401
    assert (await api.get("/api/v1/auth/me")).status_code == 200  # công khai
    await api.post("/api/v1/auth/register", json={"email": "a@x.vn", "password": "secret1"})
    assert (await api.get("/api/v1/projects")).status_code == 200
