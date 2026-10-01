import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.auth_middleware import auth_middleware
from src.api.auth_routes import auth_router
from src.api.projects_routes import projects_router
from src.api.routes import resume_videos, router
from src.api.routes3d import router3d
from src.config import get_settings

WEB_DIR = Path(__file__).parent / "web"


def _resume_projects() -> None:
    from src.services.projects import get_manager

    try:
        get_manager().resume()
    except Exception:  # không để lỗi ở đây làm hỏng lúc khởi động server
        import logging

        logging.getLogger(__name__).exception("Không xếp hàng lại được dự án")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"Starting {settings.app_name} in {settings.app_env} mode — UI: http://localhost:{settings.app_port}/")
    # Video tải lên còn dở (server tắt / --reload giữa lúc auto-label): làm tiếp ở luồng nền
    threading.Thread(target=resume_videos, name="resume-videos", daemon=True).start()
    # Dự án của end-user đang chờ / xử lý dở: xếp hàng lại
    threading.Thread(target=_resume_projects, name="resume-projects", daemon=True).start()
    yield
    print("Shutting down...")


app = FastAPI(
    title="AutoLabel 3D",
    description="Auto-label 2D + 3D (nuScenes, KITTI, LiDAR + camera, video, ảnh) + QA Agent + review by exception",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router3d, prefix="/api/v1")
app.include_router(projects_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.middleware("http")(auth_middleware)
# Cùng API duyệt, gắn theo dự án: /p/<id>/api/v1/frames ... (UI mở bằng /ui/?project=<id>)
app.include_router(router, prefix="/p/{project_id}/api/v1", include_in_schema=False)
app.include_router(router3d, prefix="/p/{project_id}/api/v1", include_in_schema=False)


@app.middleware("http")
async def revalidate_ui(request, call_next):
    # UI thay đổi thường xuyên: bắt trình duyệt hỏi lại server mỗi lần (304 nếu file không đổi),
    # tránh chạy app.js cũ sau khi cập nhật code
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/ui"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse("/ui/projects.html")


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/login", include_in_schema=False)
async def login_page():
    return RedirectResponse("/ui/login.html")


@app.get("/projects", include_in_schema=False)
async def projects_page():
    return RedirectResponse("/ui/projects.html")


@app.get("/frames", include_in_schema=False)
async def frames_page():
    return RedirectResponse("/ui/frames.html")


@app.get("/project/{project_id}/frames", include_in_schema=False)
async def project_frames_page(project_id: str):
    return RedirectResponse(f"/ui/frames.html?project={project_id}")


@app.get("/annotate", include_in_schema=False)
@app.get("/annotate/{frame_id}", include_in_schema=False)
async def annotate_page(frame_id: str = ""):
    if frame_id:
        return RedirectResponse(f"/ui/?frame={frame_id}")
    return RedirectResponse("/ui/")


@app.get("/export", include_in_schema=False)
async def export_page():
    return RedirectResponse("/ui/export.html")


@app.get("/flow", include_in_schema=False)
async def flow_page():
    return RedirectResponse("/ui/projects.html")

