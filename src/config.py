from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "AutoLabel 3D"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000"

    # Dữ liệu nuScenes: trainval blob01 (85 scene, 3376 keyframe, đủ 6 camera + LiDAR). Bảng v1.0-trainval-blob01
    # chỉ gồm các scene có file trên máy; bảng v1.0-trainval đầy đủ (850 scene) thiếu ảnh của phần lớn scene
    nuscenes_dataroot: str = "./data/nuscenes-trainval"
    nuscenes_version: str = "v1.0-trainval-blob01"

    # Nơi lưu kết quả auto-label, trạng thái review, correction log, export
    workspace_dir: str = "./data/workspace-trainval-yoloe"
    autolabel_config: str = "./configs/autolabel.yaml"
    reviewer_name: str = "annotator"

    # Dự án của end-user (trang /ui/projects.html): mỗi dự án một thư mục con
    projects_dir: str = "./data/projects"
    # Python của môi trường MMDetection3D (tools3d/setup.ps1); để trống = tự tìm .venv-mm3d trong repo
    mm3d_python: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
