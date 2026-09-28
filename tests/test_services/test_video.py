"""Chế độ Video: cắt frame mp4, auto-label bằng detector demo (không cần GPU), lan truyền trên video tải lên."""

from pathlib import Path

import pytest

from src import demo
from src.models.schemas import ReviewActionRequest
from src.services import review
from src.services import video as vs
from src.services.detectors import DetectorEnsemble
from src.services.sequence import PropagationError, WorkspaceSequenceSource, propagate_from
from src.services.store import WORKSPACE_PREFIX, WorkspaceStore
from tests.conftest import ROOT, make_frame


@pytest.fixture(scope="module")
def demo_mp4(tmp_path_factory) -> Path:
    # 3 giây ở 10 fps -> 30 frame, 6 keyframe
    return demo.render_video(tmp_path_factory.mktemp("video") / "street.mp4", seconds=3)


@pytest.fixture
def demo_config(config):
    config.detection.detectors = ["demo"]
    return config


@pytest.fixture
def labeled(tmp_path, demo_config, demo_mp4):
    store = WorkspaceStore(tmp_path / "ws")
    ensemble = DetectorEnsemble(demo_config, store.root / "cache" / "detections")
    video = vs.import_video(store, demo_config, demo_mp4, "street.mp4", ensemble.names)
    vs.process_video(store, ensemble, demo_config, video.video_id)
    return store, ensemble, video.video_id


def no_nuscenes():
    raise FileNotFoundError("không có nuScenes")


def find(frame, label, x_min=None, x_max=None):
    return [
        o
        for o in frame.objects
        if (o.review.final_label or o.label) == label
        and (x_min is None or o.bbox[0] >= x_min)
        and (x_max is None or o.bbox[0] <= x_max)
    ]


def test_import_video_cuts_frames_and_marks_keyframes(tmp_path, demo_config, demo_mp4):
    store = WorkspaceStore(tmp_path / "ws")
    video = vs.import_video(store, demo_config, demo_mp4, "Street Demo.mp4", ["demo"])
    assert video.video_id.startswith("vid-street-demo-")
    assert (video.width, video.height) == (1280, 720)
    assert len(video.timeline) == 30 and video.duration_s == pytest.approx(2.9)
    keys = [e for e in video.timeline if e.sample_token]
    assert len(keys) == 6 and [video.timeline.index(k) for k in keys] == [0, 5, 10, 15, 20, 25]
    assert all(e.path.startswith(WORKSPACE_PREFIX) and store.resolve(e.path).is_file() for e in video.timeline)
    assert store.load_video(video.video_id).status == "processing"


def test_import_video_rejects_unreadable_file(tmp_path, demo_config):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(vs.VideoError) as e:
        vs.import_video(WorkspaceStore(tmp_path / "ws"), demo_config, bad, "bad.mp4", ["demo"])
    assert e.value.code in {"VIDEO_UNREADABLE", "VIDEO_EMPTY"}


def test_process_video_labels_every_keyframe(labeled):
    store, _, vid = labeled
    video = store.load_video(vid)
    assert video.status == "ready" and video.progress == 1.0
    frames = sorted((f for f in store.list_frames() if f.scene == vid), key=lambda f: f.index)
    assert [f.frame_id for f in frames] == [f"{vid}_{k:03d}" for k in range(6)]

    f0 = frames[0]
    assert f0.camera == vs.UPLOAD_CAMERA and len(f0.sweeps) == 2  # chỉ có t+1, t+2 ở đầu video
    van = find(f0, "truck")
    assert len(van) == 1 and "CLASS_CONFLICT" in [i.code for i in van[0].qa.issues]
    far_car = [o for o in find(f0, "car") if o.bbox[2] - o.bbox[0] < 40]
    assert far_car and "LOW_CONFIDENCE" in [i.code for i in far_car[0].qa.issues]
    assert find(f0, "pedestrian", 540, 580)  # poster trên tường, người duyệt phải xoá


def test_process_video_never_overwrites_frames_a_human_opened(labeled, demo_config):
    store, ensemble, vid = labeled
    f0 = store.load_frame(f"{vid}_000")
    van = find(f0, "truck")[0]
    review.apply_action(
        f0, ReviewActionRequest(action="CHANGE_CLASS", object_id=van.object_id, label="car"), "an", demo_config
    )
    store.save_frame(f0)

    vs.process_video(store, ensemble, demo_config, vid)
    again = store.load_frame(f"{vid}_000")
    assert again.status == "editing" and not find(again, "truck")


def test_propagation_on_uploaded_video(labeled, demo_config):
    store, ensemble, vid = labeled
    f0 = store.load_frame(f"{vid}_000")
    van = find(f0, "truck")[0]
    poster = find(f0, "pedestrian", 540, 580)[0]
    for req in (
        ReviewActionRequest(action="CHANGE_CLASS", object_id=van.object_id, label="car"),
        ReviewActionRequest(action="DELETE", object_id=poster.object_id),
    ):
        review.apply_action(f0, req, "an", demo_config)
    for o in f0.objects:
        if o.review.status == "pending":
            review.apply_action(f0, ReviewActionRequest(action="KEEP", object_id=o.object_id), "an", demo_config)
    review.approve_frame(f0, "an", 20)
    store.save_frame(f0)

    # Video tải lên không cần nuScenes: timeline lấy từ record trong workspace
    source = WorkspaceSequenceSource(store, ensemble, no_nuscenes)
    resp = propagate_from(store, source, demo_config, f0.frame_id)
    assert resp.frames_updated == [f"{vid}_{k:03d}" for k in range(1, 6)]

    for k in range(1, 6):
        fk = store.load_frame(f"{vid}_{k:03d}")
        assert fk.propagated_from == f0.frame_id
        assert not find(fk, "truck"), "lớp người sửa ở keyframe phải được mang sang"
        cars = [o for o in find(fk, "car") if o.source == "propagated" and o.bbox[1] == pytest.approx(430, abs=3)]
        assert len(cars) == 1 and cars[0].track_id and cars[0].propagation.matched
        posters = [o for o in fk.objects if o.bbox[0] == pytest.approx(560, abs=4) and o.label == "pedestrian"]
        assert posters and all(p.review.action == "PROPAGATED_DELETE" for p in posters)


def test_nuscenes_scene_without_dataset_is_reported(tmp_path, demo_config):
    store = WorkspaceStore(tmp_path / "ws")
    source = WorkspaceSequenceSource(store, DetectorEnsemble(demo_config, tmp_path / "cache"), no_nuscenes)
    with pytest.raises(PropagationError) as e:
        source.timeline("scene-0061", "CAM_FRONT")
    assert e.value.code == "DATA_UNAVAILABLE" and e.value.status == 503


def test_list_videos_puts_uploads_first_then_nuscenes_scenes(labeled):
    store, _, vid = labeled
    store.save_frame(make_frame("scene-0061_000"))
    videos = vs.list_videos(store)
    assert [(v.video_id, v.source) for v in videos] == [(vid, "upload"), ("scene-0061", "nuscenes")]
    assert videos[0].n_frames == 6 and videos[0].first_frame_id == f"{vid}_000"

    detail = vs.video_detail(store, vid)
    assert [f.t for f in detail.frames] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert vs.video_detail(store, "nope") is None


def test_demo_build_is_self_contained(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(demo, "DEMO_DIR", tmp_path / "demo")
    paths = demo.build(seconds=1.5)
    store = WorkspaceStore(paths["workspace"])
    (video,) = store.list_videos()
    assert video.status == "ready" and len(store.list_frames()) == 3
    assert (tmp_path / "demo" / "street_demo.mp4").is_file()
    # Chạy lại không sinh thêm video
    demo.build(seconds=1.5)
    assert len(WorkspaceStore(paths["workspace"]).list_videos()) == 1


def test_resume_pending_finishes_videos_left_processing(tmp_path, demo_config, demo_mp4):
    # Server tắt giữa lúc auto-label: record còn "processing", mới có frame đầu
    store = WorkspaceStore(tmp_path / "ws")
    ensemble = DetectorEnsemble(demo_config, store.root / "cache" / "detections")
    video = vs.import_video(store, demo_config, demo_mp4, "street.mp4", ensemble.names)
    assert vs.resume_pending(store, ensemble, demo_config) == [video.video_id]
    assert store.load_video(video.video_id).status == "ready"
    assert len([f for f in store.list_frames() if f.scene == video.video_id]) == 6
    assert vs.resume_pending(store, ensemble, demo_config) == []


def test_reprocessing_keeps_propagated_labels(labeled, demo_config):
    store, ensemble, vid = labeled
    f0 = store.load_frame(f"{vid}_000")
    review.approve_low_risk(f0, "an")
    for o in f0.objects:
        if o.review.status == "pending":
            review.apply_action(f0, ReviewActionRequest(action="KEEP", object_id=o.object_id), "an", demo_config)
    review.approve_frame(f0, "an", 5)
    store.save_frame(f0)
    propagate_from(store, WorkspaceSequenceSource(store, ensemble, no_nuscenes), demo_config, f0.frame_id)

    vs.process_video(store, ensemble, demo_config, vid)  # vd. làm tiếp sau khi server khởi động lại
    assert store.load_frame(f"{vid}_003").propagated_from == f0.frame_id


def test_sparse_video_skips_temporal_check(tmp_path, demo_config, demo_mp4):
    # Video chỉ có keyframe 2 Hz: frame lân cận cách 0.5 s, không dùng cho check temporal (tránh FLICKER sai)
    demo_config.video.track_fps = 2.0
    store = WorkspaceStore(tmp_path / "ws")
    ensemble = DetectorEnsemble(demo_config, store.root / "cache" / "detections")
    video = vs.import_video(store, demo_config, demo_mp4, "street.mp4", ensemble.names)
    vs.process_video(store, ensemble, demo_config, video.video_id)
    frames = [f for f in store.list_frames() if f.scene == video.video_id]
    assert len(frames) == 6 and all(not f.sweeps for f in frames)
    assert not any(i.code == "FLICKER" for f in frames for o in f.objects for i in o.qa.issues)
