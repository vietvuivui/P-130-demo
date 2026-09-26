from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api import routes
from src.main import app
from src.models.qa_config import AutoLabelConfig, load_autolabel_config
from src.models.schemas import FrameRecord, ImageInfo, LabelObject, QAIssue, QAResult, SweepInfo
from src.services.store import WorkspaceStore

ROOT = Path(__file__).resolve().parents[1]
INTRINSIC = [[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]]


@pytest.fixture
def config() -> AutoLabelConfig:
    return load_autolabel_config(ROOT / "configs" / "autolabel.yaml")


def make_object(
    oid: str, bbox, label="car", score=0.9, level="low", risk=0.1, issues=(), source="model"
) -> LabelObject:
    return LabelObject(
        object_id=oid,
        bbox=list(bbox),
        label=label,
        score=score,
        source=source,
        qa=QAResult(
            risk=risk,
            level=level,
            issues=[QAIssue(code=c, group="detection", message=c) for c in issues],
        ),
    )


def make_frame(frame_id="scene-0001_000", objects=None) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        sample_token="tok_" + frame_id,
        scene=frame_id.split("_")[0],
        index=int(frame_id.split("_")[1]),
        camera="CAM_FRONT",
        image=ImageInfo(sd_token="sd0", path="samples/CAM_FRONT/key.jpg", timestamp=1_000_000),
        intrinsic=INTRINSIC,
        sweeps=[SweepInfo(offset=-1, sd_token="sd-1", path="sweeps/CAM_FRONT/prev.jpg", timestamp=900_000)],
        detectors=["yolo_world"],
        frame_risk=max((o.qa.risk for o in objects or []), default=0.0),
        objects=objects
        if objects is not None
        else [
            make_object("1", [100, 400, 300, 520], score=0.92, level="low", risk=0.05),
            make_object("2", [700, 420, 760, 560], label="pedestrian", score=0.8, level="low", risk=0.12),
            make_object(
                "3",
                [900, 300, 1000, 330],
                label="barrier",
                score=0.39,
                level="high",
                risk=0.81,
                issues=("NO_LIDAR_SUPPORT", "FLICKER"),
            ),
        ],
    )


@pytest.fixture
def store(tmp_path) -> WorkspaceStore:
    s = WorkspaceStore(tmp_path / "workspace")
    s.save_frame(make_frame())
    return s


@pytest.fixture
def dataroot(tmp_path) -> Path:
    img = tmp_path / "data" / "samples" / "CAM_FRONT" / "key.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff\xd9")
    return tmp_path / "data"


@pytest_asyncio.fixture
async def client(store, dataroot, config):
    """Client HTTP với workspace tạm, không đụng dữ liệu thật."""
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_dataroot] = lambda: dataroot
    app.dependency_overrides[routes.get_config] = lambda: config
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
