"""Tải file lớn từng phần (PUT /projects/uploads/{id}) rồi tạo dự án bằng upload_id."""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.services import projects
from src.services.projects import ProjectManager
from tests.test_services.test_projects import demo_image


@pytest.fixture
def manager(tmp_path, config, monkeypatch):
    cfg = config.model_copy(deep=True)
    cfg.detection.detectors = ["demo"]
    m = ProjectManager(tmp_path / "projects", lambda: cfg, mm3d_python=str(tmp_path / "no-mm3d"))
    monkeypatch.setattr(projects, "_manager", m)
    return m


@pytest_asyncio.fixture
async def api(manager):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_chunked_upload_creates_project(api, manager, tmp_path):
    img = tmp_path / "anh.jpg"
    demo_image(img)
    data = img.read_bytes()
    uid = uuid.uuid4().hex
    url = f"/api/v1/projects/uploads/{uid}"
    half = len(data) // 2
    r = await api.put(url, params={"name": "anh.jpg", "offset": 0}, content=data[:half])
    assert r.status_code == 200 and r.json()["size"] == half
    # gửi lại phần đã có (mất phản hồi) -> 409 kèm số byte máy chủ đang có để gửi tiếp
    r = await api.put(url, params={"name": "anh.jpg", "offset": 0}, content=data[:half])
    assert r.status_code == 409 and r.json()["detail"]["size"] == half
    r = await api.put(url, params={"name": "anh.jpg", "offset": half}, content=data[half:])
    assert r.status_code == 200 and r.json()["size"] == len(data)

    r = await api.post("/api/v1/projects", data={"upload_id": uid, "name": "Lớn"})
    assert r.status_code == 200, r.text
    p = manager.get(r.json()["id"])
    assert p.uploads == ["anh.jpg"]
    assert (manager.dir(p.id) / "upload" / "anh.jpg").read_bytes() == data


@pytest.mark.asyncio
async def test_chunked_upload_rejects_bad_input(api, manager):
    r = await api.put(f"/api/v1/projects/uploads/{uuid.uuid4().hex}", params={"name": ".."}, content=b"x")
    assert r.status_code == 422
    r = await api.put(f"/api/v1/projects/uploads/{'z' * 32}", params={"name": "a.jpg"}, content=b"x")
    assert r.status_code == 422
    r = await api.post("/api/v1/projects", data={"upload_id": uuid.uuid4().hex})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "NO_FILES"
    r = await api.post("/api/v1/projects", data={"name": "rỗng"})
    assert r.status_code == 422
