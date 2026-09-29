"""Chế độ Video: video tải lên (mp4) và danh sách video cho UI.

Một video được xử lý giống một scene nuScenes để dùng lại toàn bộ pipeline và phần lan truyền:
- cắt frame ở `track_fps` (mặc định 10 fps) — tương đương sweep 12Hz, dùng để tracking;
- cứ `track_fps / label_fps` frame thì có một keyframe (mặc định 2 fps) — tương đương keyframe 2Hz, có
  FrameRecord để người gán nhãn; sweep t-2..t+2 quanh keyframe được detect cùng lúc cho QA temporal.

Frame lưu trong workspace: videos/<video_id>/frames/00012.jpg, đường dẫn ghi dạng '@workspace/...'.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import uuid
from collections import defaultdict
from pathlib import Path

from src.models.qa_config import AutoLabelConfig
from src.models.schemas import (
    FrameRecord,
    ImageInfo,
    TimelineEntry,
    VideoDetail,
    VideoFrame,
    VideoRecord,
    VideoSummary,
)
from src.services.detectors import DetectorEnsemble
from src.services.pipeline import label_keyframe
from src.services.review import now_iso, summarize
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
UPLOAD_CAMERA = "video"
# Mỗi lúc chỉ xử lý một video: detector (YOLOE, YOLO-World...) không an toàn khi hai luồng cùng gọi, và một GPU cũng
# không nhanh hơn khi chạy song song
_PROCESS_LOCK = threading.Lock()


class VideoError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


def cv_path(path: Path) -> str:
    """OpenCV trên Windows không mở được đường dẫn có ký tự ngoài ASCII (vd. C:\\Users\\Kiên\\...). Khi đó thử
    đường dẫn tương đối so với thư mục hiện tại, thường chỉ còn phần ASCII."""
    s = str(path)
    if s.isascii():
        return s
    try:
        rel = os.path.relpath(path)
    except ValueError:  # khác ổ đĩa
        return s
    return rel if rel.isascii() else s


def write_jpeg(path: Path, image, quality: int = 90) -> None:
    """Ghi ảnh qua imencode + tofile (đường dẫn Unicode vẫn ghi được), báo lỗi thay vì lặng lẽ bỏ qua."""
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("FRAME_WRITE_FAILED", f"Không mã hoá được frame {path.name}")
    buf.tofile(str(path))


def new_video_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem).strip("-").lower()[:24] or "video"
    return f"vid-{slug}-{uuid.uuid4().hex[:6]}"


def extract_frames(
    src: Path, store: WorkspaceStore, video: VideoRecord, max_seconds: float, max_frames: int | None = None
) -> VideoRecord:
    """Cắt frame bằng OpenCV, ghi vào workspace, điền timeline/kích thước/thời lượng cho record."""
    import cv2

    cap = cv2.VideoCapture(cv_path(src))
    if not cap.isOpened():
        raise VideoError("VIDEO_UNREADABLE", "Không đọc được file video (thử mp4 H.264)")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / video.track_fps))
    key_every = max(1, round(video.track_fps / video.label_fps))
    out_dir = store.video_dir(video.video_id) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline: list[TimelineEntry] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or idx / src_fps > max_seconds or (max_frames and len(timeline) >= max_frames):
                break
            if idx % step == 0:
                n = len(timeline)
                name = f"{n:05d}.jpg"
                write_jpeg(out_dir / name, frame)
                if not video.width:
                    video.height, video.width = frame.shape[:2]
                timeline.append(
                    TimelineEntry(
                        sd_token=f"{video.video_id}-{n:05d}",
                        timestamp=int(round(idx / src_fps * 1e6)),
                        sample_token=f"{video.video_id}-k{n // key_every:03d}" if n % key_every == 0 else None,
                        path=f"{WORKSPACE_PREFIX}videos/{video.video_id}/frames/{name}",
                    )
                )
            idx += 1
    finally:
        cap.release()
    if not timeline:
        raise VideoError("VIDEO_EMPTY", "Video không có frame nào đọc được")
    video.timeline = timeline
    video.duration_s = round((timeline[-1].timestamp - timeline[0].timestamp) / 1e6, 2)
    return video


def import_video(
    store: WorkspaceStore, config: AutoLabelConfig, src: Path, name: str, detectors: list[str],
    max_frames: int | None = None,
) -> VideoRecord:  # fmt: skip
    """Tạo record + cắt frame. Việc detect/QA (nặng) chạy riêng bằng process_video. max_frames: chỉ cắt N frame đầu."""
    cfg = config.video
    video = VideoRecord(
        video_id=new_video_id(name),
        name=Path(name).name,
        track_fps=cfg.track_fps,
        label_fps=cfg.label_fps,
        detectors=detectors,
        created_at=now_iso(),
        message="Đang cắt frame",
    )
    try:
        extract_frames(src, store, video, cfg.max_seconds, max_frames)
    except Exception:
        shutil.rmtree(store.video_dir(video.video_id), ignore_errors=True)  # không để lại frame dở dang
        raise
    video.message = "Đang auto-label"
    store.save_video(video)
    return video


def import_images(
    store: WorkspaceStore, config: AutoLabelConfig, files: list[Path], name: str, detectors: list[str],
    sequential: bool = False,
) -> VideoRecord:  # fmt: skip
    """Bộ ảnh -> record giống video. sequential=True: ảnh là chuỗi liên tục (như frame video 10 fps, keyframe 2 fps, lan
    truyền được); False: ảnh rời, mỗi ảnh một keyframe, không dùng ảnh lân cận cho QA temporal và không lan truyền."""
    import cv2
    import numpy as np

    if not files:
        raise VideoError("NO_IMAGES", "Không có ảnh nào (jpg, png)")
    cfg = config.video
    video = VideoRecord(
        video_id=new_video_id(name), name=Path(name).name, source="upload" if sequential else "images",
        track_fps=cfg.track_fps, label_fps=cfg.label_fps if sequential else cfg.track_fps, detectors=detectors,
        created_at=now_iso(), message="Đang chuẩn bị ảnh",
    )  # fmt: skip
    key_every = max(1, round(cfg.track_fps / cfg.label_fps)) if sequential else 1
    step_s = 1 / cfg.track_fps if sequential else 10.0  # ảnh rời cách nhau 10 s: QA temporal không lấy ảnh bên cạnh
    out_dir = store.video_dir(video.video_id) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for n, f in enumerate(sorted(files, key=lambda p: p.name)):
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if not video.width:
                video.height, video.width = img.shape[:2]
            elif img.shape[:2] != (video.height, video.width):  # khác cỡ: co giữ tỉ lệ, viền đen
                scale = min(video.width / img.shape[1], video.height / img.shape[0])
                small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
                canvas = np.zeros((video.height, video.width, 3), np.uint8)
                y0, x0 = (video.height - small.shape[0]) // 2, (video.width - small.shape[1]) // 2
                canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
                img = canvas
            k = len(video.timeline)
            fname = f"{k:05d}.jpg"
            write_jpeg(out_dir / fname, img)
            video.timeline.append(TimelineEntry(
                sd_token=f"{video.video_id}-{k:05d}", timestamp=int(round(k * step_s * 1e6)),
                sample_token=f"{video.video_id}-k{k // key_every:03d}" if k % key_every == 0 else None,
                path=f"{WORKSPACE_PREFIX}videos/{video.video_id}/frames/{fname}",
            ))  # fmt: skip
    except Exception:
        shutil.rmtree(store.video_dir(video.video_id), ignore_errors=True)
        raise
    if not video.timeline:
        shutil.rmtree(store.video_dir(video.video_id), ignore_errors=True)
        raise VideoError("NO_IMAGES", "Không đọc được ảnh nào")
    video.duration_s = round(video.timeline[-1].timestamp / 1e6, 2)
    video.message = "Đang auto-label"
    store.save_video(video)
    return video


def _image(entry: TimelineEntry, video: VideoRecord) -> ImageInfo:
    return ImageInfo(
        sd_token=entry.sd_token, path=entry.path, timestamp=entry.timestamp, width=video.width, height=video.height
    )


def process_video(store: WorkspaceStore, ensemble: DetectorEnsemble, config: AutoLabelConfig, video_id: str) -> None:
    """Auto-label + QA cho mọi keyframe của video tải lên. Keyframe đã có frame thì bỏ qua (người có thể đã mở, hoặc
    đã nhận nhãn lan truyền), nên chạy lại sau khi server khởi động lại chỉ làm tiếp phần còn thiếu."""
    with _PROCESS_LOCK:
        _process_video(store, ensemble, config, video_id)


def _save_progress(store: WorkspaceStore, video: VideoRecord) -> None:
    try:
        store.save_video(video)
    except OSError:  # chỉ là cập nhật tiến độ: lỗi ghi tạm thời không được làm dừng cả video
        log.warning("Không ghi được tiến độ video %s", video.video_id, exc_info=True)


def _process_video(store: WorkspaceStore, ensemble: DetectorEnsemble, config: AutoLabelConfig, video_id: str) -> None:
    video = store.load_video(video_id)
    if video is None:
        return
    try:
        keys = [i for i, e in enumerate(video.timeline) if e.sample_token]
        # Camera không có calibration: intrinsic danh nghĩa (tiêu cự = chiều rộng ảnh); không có LiDAR nên
        # các check LiDAR tự bỏ qua
        intrinsic = [
            [float(video.width), 0.0, video.width / 2],
            [0.0, float(video.width), video.height / 2],
            [0.0, 0.0, 1.0],
        ]
        for k, pos in enumerate(keys):
            frame_id = f"{video_id}_{k:03d}"
            if store.load_frame(frame_id, copy=False) is None:
                t0 = video.timeline[pos].timestamp
                sweeps = {
                    o: _image(video.timeline[pos + o], video)
                    for o in config.sweep_offsets
                    if 0 <= pos + o < len(video.timeline)
                    and abs(video.timeline[pos + o].timestamp - t0) <= config.video.max_sweep_gap_s * 1e6
                }
                record = label_keyframe(
                    ensemble,
                    config,
                    frame_id=frame_id,
                    sample_token=video.timeline[pos].sample_token,
                    scene=video_id,
                    index=k,
                    camera=UPLOAD_CAMERA,
                    image=_image(video.timeline[pos], video),
                    sweeps=sweeps,
                    image_file=store.resolve,
                    intrinsic=intrinsic,
                )
                store.save_frame(record)
            video.progress = round((k + 1) / len(keys), 3)
            _save_progress(store, video)
        video.status, video.message = "ready", None
    except Exception as e:  # lỗi model/IO: ghi vào record để UI hiện, không làm sập server
        log.exception("Xử lý video %s lỗi", video_id)
        video.status, video.message = "error", f"{type(e).__name__}: {e}"
    store.save_video(video)


def resume_pending(store: WorkspaceStore, ensemble: DetectorEnsemble, config: AutoLabelConfig) -> list[str]:
    """Làm tiếp video còn "processing" (server tắt / reload giữa chừng). Gọi khi server khởi động."""
    pending = [v.video_id for v in store.list_videos() if v.status == "processing"]
    for vid in pending:
        log.info("Làm tiếp video %s", vid)
        process_video(store, ensemble, config, vid)
    return pending


# ---------------------------------------------------------------------------
# Danh sách video cho UI: scene nuScenes (suy từ frame) + video tải lên (có record)
# ---------------------------------------------------------------------------


def video_summary(video_id: str, frames: list[FrameRecord], record: VideoRecord | None) -> VideoSummary:
    frames = sorted(frames, key=lambda f: f.index)
    ts = [f.image.timestamp for f in frames]
    return VideoSummary(
        video_id=video_id,
        name=record.name if record else video_id,
        source=record.source if record else "nuscenes",
        status=record.status if record else "ready",
        progress=record.progress if record else 1.0,
        message=record.message if record else None,
        n_frames=len(frames),
        approved=sum(f.status == "approved" for f in frames),
        editing=sum(f.status == "editing" for f in frames),
        propagated=sum(f.propagated_from is not None and f.status == "auto" for f in frames),
        pending_objects=sum(o.review.status == "pending" for f in frames for o in f.objects),
        duration_s=record.duration_s if record else (round((ts[-1] - ts[0]) / 1e6, 2) if ts else 0.0),
        first_frame_id=frames[0].frame_id if frames else None,
    )


def list_videos(store: WorkspaceStore) -> list[VideoSummary]:
    groups: dict[str, list[FrameRecord]] = defaultdict(list)
    for f in store.list_frames():
        groups[f.scene].append(f)
    records = {v.video_id: v for v in store.list_videos()}
    out = [video_summary(vid, groups.get(vid, []), records.get(vid)) for vid in sorted(set(groups) | set(records))]
    # Video tải lên mới nhất lên đầu, sau đó scene nuScenes theo tên
    uploads = sorted(
        (v for v in out if v.source == "upload"), key=lambda v: records[v.video_id].created_at or "", reverse=True
    )
    return uploads + [v for v in out if v.source == "nuscenes"]


def video_detail(store: WorkspaceStore, video_id: str) -> VideoDetail | None:
    frames = [f for f in store.list_frames() if f.scene == video_id]
    record = store.load_video(video_id)
    if not frames and record is None:
        return None
    frames.sort(key=lambda f: f.index)
    t0 = frames[0].image.timestamp if frames else 0
    items = [
        VideoFrame(**summarize(f).model_dump(), timestamp=f.image.timestamp, t=round((f.image.timestamp - t0) / 1e6, 2))
        for f in frames
    ]
    return VideoDetail(**video_summary(video_id, frames, record).model_dump(), frames=items)
