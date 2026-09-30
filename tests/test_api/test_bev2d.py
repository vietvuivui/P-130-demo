"""BEV cho chế độ Ảnh / Video: thông số homography và ảnh camera chiếu xuống mặt đường."""

import io

import numpy as np
import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_bev_meta_nominal_camera(client, store):
    fid = store.list_frames()[0].frame_id
    r = await client.get(f"/api/v1/frames/{fid}/bev/meta")
    assert r.status_code == 200
    m = r.json()
    # không có bảng nuScenes trong test -> camera giả định cao 1.5 m, nhìn thẳng
    assert m["estimated"] is True and m["camera_height"] == pytest.approx(1.5)
    frame = (await client.get(f"/api/v1/frames/{fid}")).json()
    intr = np.asarray(frame["intrinsic"])
    uvw = np.asarray(m["homography"]) @ np.array([10.0, 0.0, 1.0])  # 10 m thẳng phía trước, trên mặt đường
    u, v = uvw[:2] / uvw[2]
    assert u == pytest.approx(intr[0, 2]) and v == pytest.approx(intr[1, 2] + intr[1, 1] * 1.5 / 10)
    # điểm lệch trái (y > 0) nằm bên trái ảnh
    uvw = np.asarray(m["homography"]) @ np.array([10.0, 2.0, 1.0])
    assert uvw[0] / uvw[2] < intr[0, 2]


@pytest.mark.asyncio
async def test_bev_image(client, store, dataroot):
    frame = store.list_frames()[0]
    img = np.zeros((frame.image.height, frame.image.width, 3), np.uint8)
    img[frame.image.height // 2 + 10 :] = [30, 200, 30]  # mặt đường (dưới đường chân trời) màu xanh
    path = dataroot / frame.image.path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="JPEG")
    r = await client.get(f"/api/v1/frames/{frame.frame_id}/bev")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    bev = np.asarray(Image.open(io.BytesIO(r.content)))
    assert bev.shape == (500, 400, 4)  # 50 m phía trước x 40 m ngang, 0.1 m/px
    assert bev[400, 200, 3] > 0 and bev[400, 200, 1] > 150  # 10 m thẳng trước: thấy mặt đường
    assert bev[499, 0, 3] == 0  # sát xe, lệch 20 m: ngoài góc nhìn
