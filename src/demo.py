"""Demo chạy không cần GPU, không cần nuScenes (`python -m src.cli run` của MVP cần cả hai).

    python -m src.demo                 # sinh video demo, auto-label, mở UI ở http://localhost:8000
    python -m src.demo --no-serve      # chỉ sinh dữ liệu
    python -m src.demo --reset         # xoá dữ liệu demo cũ rồi sinh lại

Sinh một video đường phố tổng hợp (data/demo/street_demo.mp4) rồi cho nó đi qua **đúng đường xử lý của video
tải lên**: cắt frame 10 fps, keyframe 2 fps, detect, QA Agent. Detector là bản demo tìm vật thể theo màu
(src/services/detectors/demo.py), nên chạy được trên máy bất kỳ. Dữ liệu demo nằm riêng ở data/demo/, không
đụng workspace thật. Muốn thử nút "Tải lên mp4" trên UI thì chọn lại chính file street_demo.mp4.

Tình huống có sẵn để demo (xem docs/demo-script.md):
- xe đỏ đi qua sau biển quảng cáo (bị che dần rồi che hẳn) -> box bị cắt (ASPECT_RATIO_ABNORMAL); che hẳn thì track
  dừng, xe hiện lại là object mới cần duyệt
- xe van cam bị detector nhận là truck, phân vân car     -> sửa lớp ở keyframe, lan truyền mang lớp đúng
- poster hình người trên tường bị nhận là pedestrian      -> xoá ở keyframe, frame sau tự xoá
- xe đạp xuất hiện giữa video                             -> object mới, không có ở keyframe đầu
- xe xanh nhỏ ở xa                                        -> score thấp (LOW_CONFIDENCE)
- vệt đỏ loé lên ở vài frame                              -> FLICKER
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.services.detectors.demo import DEMO_PALETTE

DEMO_DIR = Path("data/demo")
W, H, FPS = 1280, 720, 10
SECONDS = 16


def _bgr(name: str) -> tuple[int, int, int]:
    r, g, b = DEMO_PALETTE[name]["rgb"]
    return (b, g, r)


def _vehicle(img, x: float, y: float, w: int, h: int, color) -> None:
    import cv2

    x1, y1 = int(round(x)), int(round(y))
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
    # bánh xe tối màu nằm ngoài thân xe để không làm vỡ vùng màu
    for cx in (x1 + w // 5, x1 + 4 * w // 5):
        cv2.circle(img, (cx, y1 + h + 6), max(5, h // 7), (25, 25, 25), -1)


def render_frame(i: int) -> np.ndarray:
    import cv2

    img = np.zeros((H, W, 3), np.uint8)
    # trời, nhà, vỉa hè, đường (màu xám/nhạt, không trùng bảng màu vật thể)
    for y in range(0, 300):
        c = 205 - y // 6
        img[y, :] = (c, c - 10, c - 30)
    for bx, bw, bh, c in (
        (0, 260, 250, 120),
        (260, 200, 200, 140),
        (460, 300, 270, 110),
        (760, 220, 220, 130),
        (980, 300, 260, 115),
    ):
        cv2.rectangle(img, (bx, 300 - bh), (bx + bw, 300), (c, c, c), -1)
    cv2.rectangle(img, (0, 300), (W, 400), (150, 150, 145), -1)
    cv2.rectangle(img, (0, 400), (W, H), (70, 72, 75), -1)
    for x in range(0, W, 120):
        cv2.rectangle(img, (x, 555), (x + 60, 563), (225, 225, 225), -1)

    # poster hình người trên tường (detector sẽ nhận nhầm là pedestrian)
    cv2.rectangle(img, (560, 110), (600, 205), _bgr("pedestrian"), -1)
    cv2.rectangle(img, (548, 100), (612, 215), (90, 90, 90), 3)

    # xe xanh đỗ sát lề (chắc chắn) và xe xanh nhỏ ở xa cuối đường (score thấp); không nằm trên đường đi của xe khác
    _vehicle(img, 1000, 312, 200, 74, _bgr("car_blue"))
    _vehicle(img, 330, 404, 20, 11, _bgr("car_blue"))

    # xe van cam, làn xa, chạy phải -> trái
    _vehicle(img, 1150 - 6 * i, 430, 170, 70, _bgr("van_orange"))
    # người đi bộ trên vỉa hè
    px = 120 + 2.2 * i
    cv2.rectangle(img, (int(px), 300), (int(px) + 34, 390), _bgr("pedestrian"), -1)
    # xe đạp vào từ bên phải ở giây thứ 6
    if i >= 60:
        cx = 1300 - 9 * (i - 60)
        cv2.rectangle(img, (int(cx), 470), (int(cx) + 60, 530), _bgr("cyclist"), -1)
    # xe đỏ, làn gần, chạy trái -> phải, đi qua sau biển quảng cáo
    _vehicle(img, 40 + 10 * i, 600, 180, 80, _bgr("car_red"))
    # biển quảng cáo đứng trước làn gần (che hẳn xe đỏ khoảng 5 frame)
    cv2.rectangle(img, (660, 560), (890, 720), (40, 70, 40), -1)
    cv2.rectangle(img, (672, 572), (878, 700), (70, 110, 70), -1)
    # vệt đỏ loé lên (phản chiếu) ở vài frame
    if i in (20, 70, 120):
        cv2.rectangle(img, (230, 470), (262, 492), _bgr("car_red"), -1)
    return img


def render_video(path: Path, seconds: float = SECONDS) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    from src.services.video import cv_path

    writer = cv2.VideoWriter(cv_path(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV không ghi được video vào {path}")
    for i in range(int(seconds * FPS)):
        writer.write(render_frame(i))
    writer.release()
    return path


def demo_config_path() -> Path:
    """Bản sao configs/autolabel.yaml nhưng dùng detector demo."""
    cfg = yaml.safe_load(Path("configs/autolabel.yaml").read_text(encoding="utf-8"))
    cfg["detection"]["detectors"] = ["demo"]
    out = DEMO_DIR / "autolabel.demo.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def build(reset: bool = False, seconds: float = SECONDS) -> dict:
    from src.models.qa_config import load_autolabel_config
    from src.services.detectors import DetectorEnsemble
    from src.services.store import WorkspaceStore
    from src.services.video import import_video, process_video

    if reset and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    workspace = DEMO_DIR / "workspace"
    config_path = demo_config_path()
    store = WorkspaceStore(workspace)
    if store.list_videos():
        print(f"Đã có dữ liệu demo ở {workspace} (chạy --reset để sinh lại)")
        return {"workspace": workspace, "config": config_path}

    config = load_autolabel_config(config_path)
    video_path = render_video(DEMO_DIR / "street_demo.mp4", seconds)
    ensemble = DetectorEnsemble(config, workspace / "cache" / "detections")
    video = import_video(store, config, video_path, "street_demo.mp4", ensemble.names)
    process_video(store, ensemble, config, video.video_id)
    video = store.load_video(video.video_id)
    print(
        f"Video demo: {video_path} -> {video.video_id}: {len(video.timeline)} frame, "
        f"{sum(e.sample_token is not None for e in video.timeline)} keyframe, trạng thái {video.status}"
    )
    return {"workspace": workspace, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.demo", description="Demo không cần GPU")
    parser.add_argument("--reset", action="store_true", help="Xoá data/demo rồi sinh lại")
    parser.add_argument("--no-serve", action="store_true", help="Chỉ sinh dữ liệu, không mở server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=SECONDS, help="Độ dài video demo")
    args = parser.parse_args()

    paths = build(args.reset, args.seconds)
    if args.no_serve:
        return
    # Trỏ app sang workspace + config demo trước khi import app
    os.environ["WORKSPACE_DIR"] = str(paths["workspace"])
    os.environ["AUTOLABEL_CONFIG"] = str(paths["config"])
    os.environ.setdefault("REVIEWER_NAME", "demo")
    from src.config import get_settings

    get_settings.cache_clear()
    import uvicorn

    print(f"Mở http://localhost:{args.port} — chọn chế độ 🎞 Video trên thanh trên cùng")
    uvicorn.run("src.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
