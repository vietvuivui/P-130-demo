from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import get_settings

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"Starting {settings.app_name} in {settings.app_env} mode — UI: http://localhost:{settings.app_port}/")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AutoLabel 2D",
    description="Auto-label 2D trên nuScenes + QA Agent đa tín hiệu + review by exception",
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
app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse("/ui/")


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
