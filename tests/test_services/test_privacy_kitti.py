"""FR-03 làm mờ trước khi hiển thị, FR-17 xuất KITTI."""

import math
import zipfile

import numpy as np
import pytest
from PIL import Image

from src.models.qa_config import PrivacyCfg
from src.services import privacy
from src.services.export_kitti import R_VELO_FROM_NUSC, box_corners, export_kitti
from tests.test_api.test_routes3d import store3d  # noqa: F401 — fixture


def test_blur_only_regions_and_cache(tmp_path):
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    src = tmp_path / "a.jpg"
    Image.fromarray(img).save(src, quality=100)
    calls = []

    def fake(im, cfg):
        calls.append(1)
        return [(100, 50, 160, 80, "plate")]

    cfg = PrivacyCfg(pad=0.0)
    out = privacy.anonymized_path(tmp_path / "ws", src, cfg, detector=fake)
    assert out != src and out.parent.name == "anon"
    a, b = np.asarray(Image.open(src), float), np.asarray(Image.open(out), float)
    inside = np.abs(a[55:75, 105:155] - b[55:75, 105:155]).mean()
    outside = np.abs(a[120:190, 10:90] - b[120:190, 10:90]).mean()
    assert inside > 30 and inside > 4 * outside  # vùng biển số mất chi tiết, phần còn lại chỉ lệch do nén jpeg
    assert privacy.anonymized_path(tmp_path / "ws", src, cfg, detector=fake) == out and len(calls) == 1  # cache
    assert privacy.anonymized_path(tmp_path / "ws", src, PrivacyCfg(enabled=False), detector=fake) == src
    # detector lỗi / ảnh hỏng -> gửi ảnh gốc, không làm hỏng việc xem
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xd9")
    assert privacy.anonymized_path(tmp_path / "ws", bad, cfg, detector=fake) == bad


def _parse_label(line):
    p = line.split()
    return p[0], [float(v) for v in p[4:8]], [float(v) for v in p[8:15]]


@pytest.mark.asyncio
async def test_kitti_export_geometry(client, store3d, dataroot):  # noqa: F811
    img = dataroot / "samples" / "CAM_FRONT" / "x.jpg"
    Image.new("RGB", (1600, 900), (80, 80, 80)).save(img)
    base = "/api/v1/3d/frames/pointpillars/scene-0035_000"
    assert (await client.post("/api/v1/3d/export-kitti", params={"model": "pointpillars"})).status_code == 409
    for oid in ("1", "2", "3"):
        await client.post(f"{base}/actions", json={"action": "KEEP", "object_id": oid})
    await client.post(f"{base}/approve", json={})
    r = await client.post("/api/v1/3d/export-kitti", params={"model": "pointpillars"})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["frames"] == 1 and info["labels"] == 3
    z = await client.get(info["url"])
    assert z.status_code == 200
    zp = store3d.exports_dir / info["file"]
    with zipfile.ZipFile(zp) as zf:
        names = set(zf.namelist())
        assert {"training/image_2/000000.png", "training/velodyne/000000.bin", "training/calib/000000.txt",
                "training/label_2/000000.txt", "ImageSets/train.txt", "frames.csv"} <= names  # fmt: skip
        lines = zf.read("training/label_2/000000.txt").decode().strip().splitlines()
        calib = dict(line.split(": ", 1) for line in zf.read("training/calib/000000.txt").decode().splitlines())
    # dựng lại box từ nhãn KITTI theo quy ước KITTI, đưa về hệ LiDAR nuScenes, so với box gốc
    tr = np.array(calib["Tr_velo_to_cam"].split(), float).reshape(3, 4)
    frame = store3d.load_frame3d("pointpillars", "scene-0035_000")
    for line, o in zip(lines, frame.objects, strict=True):
        name, bbox, (h, w, length, x, y, z_, ry) = _parse_label(line)
        assert name == o.label and bbox[2] > bbox[0]
        center_cam = np.array([x, y - h / 2, z_])
        velo = tr[:, :3].T @ (center_cam - tr[:, 3])
        nusc = R_VELO_FROM_NUSC.T @ velo
        assert nusc == pytest.approx(o.box.center, abs=0.02)
        assert [w, length, h] == pytest.approx(o.box.size, abs=0.01)
        # hướng: vector (cos ry, 0, -sin ry) trong hệ camera = hướng đầu xe
        d_cam = np.array([math.cos(ry), 0, -math.sin(ry)])
        d_nusc = R_VELO_FROM_NUSC.T @ (tr[:, :3].T @ d_cam)
        assert math.atan2(d_nusc[1], d_nusc[0]) == pytest.approx(o.box.yaw, abs=0.01)
    assert len(box_corners([0, 0, 0], [2, 4, 1.5], 0.0)) == 8


def test_export_kitti_skips_rejected(store3d, tmp_path):  # noqa: F811
    f = store3d.load_frame3d("pointpillars", "scene-0035_000")
    f.status, f.reject_reason = "rejected", "sai"
    store3d.save_frame3d(f)
    with pytest.raises(ValueError):
        export_kitti(store3d, tmp_path, "v1.0-mini", "pointpillars", tmp_path / "k.zip", include_pending=True)
