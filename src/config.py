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
    app_name: str = "AutoLabel 2D"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000"

    # Dữ liệu nuScenes
    nuscenes_dataroot: str = "./v1.0-mini-001"
    nuscenes_version: str = "v1.0-mini"

    # Nơi lưu kết quả auto-label, trạng thái review, correction log, export
    workspace_dir: str = "./data/workspace"
    autolabel_config: str = "./configs/autolabel.yaml"
    reviewer_name: str = "annotator"


@lru_cache
def get_settings() -> Settings:
    return Settings()
