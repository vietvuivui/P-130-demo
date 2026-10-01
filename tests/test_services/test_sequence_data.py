"""Timeline camera từ bảng nuScenes và đọc detection chỉ từ cache (bảng nuScenes giả, không cần dataset)."""

import json

from src.models.schemas import Detection
from src.services.detectors import DetectorEnsemble
from src.services.nuscenes_data import NuScenesMini


def write_tables(root):
    """2 scene; scene A có 2 keyframe, giữa là 2 sweep; chuỗi sample_data nối liền sang scene B."""
    t = root / "v1.0-mini"
    t.mkdir(parents=True)
    tables = {
        "scene": [
            {"token": "scA", "name": "scene-A", "first_sample_token": "sA0"},
            {"token": "scB", "name": "scene-B", "first_sample_token": "sB0"},
        ],
        "sample": [
            {"token": "sA0", "scene_token": "scA", "next": "sA1"},
            {"token": "sA1", "scene_token": "scA", "next": ""},
            {"token": "sB0", "scene_token": "scB", "next": ""},
        ],
        "sensor": [{"token": "cam", "channel": "CAM_FRONT"}],
        "calibrated_sensor": [{"token": "cs", "sensor_token": "cam"}],
        "sample_data": [
            {"token": "k0", "sample_token": "sA0", "is_key_frame": True, "timestamp": 0, "next": "w1"},
            {"token": "w1", "sample_token": "sA0", "is_key_frame": False, "timestamp": 83, "next": "w2"},
            {"token": "w2", "sample_token": "sA1", "is_key_frame": False, "timestamp": 166, "next": "k1"},
            {"token": "k1", "sample_token": "sA1", "is_key_frame": True, "timestamp": 250, "next": "kB"},
            {"token": "kB", "sample_token": "sB0", "is_key_frame": True, "timestamp": 999, "next": ""},
        ],
    }
    for row in tables["sample_data"]:
        row.update(calibrated_sensor_token="cs", filename=f"x/{row['token']}.jpg")
    for name, rows in tables.items():
        (t / f"{name}.json").write_text(json.dumps(rows))


def test_camera_timeline_stays_inside_scene(tmp_path):
    write_tables(tmp_path)
    tl = NuScenesMini(tmp_path).camera_timeline("scene-A", "CAM_FRONT")
    assert [im.sd_token for im in tl] == ["k0", "w1", "w2", "k1"]
    assert [im.is_keyframe for im in tl] == [True, False, False, True]
    assert tl[3].sample_token == "sA1" and tl[1].sample_token is None
    assert tl[2].path == "x/w2.jpg"


def test_load_cached_reads_only_cache(tmp_path, config):
    ens = DetectorEnsemble(config, tmp_path / "cache", ["yolo_world"])
    assert ens.load_cached("sd1") is None  # chưa chạy model cho ảnh này, và không tự chạy
    cache = ens._cache_file("yolo_world", "sd1")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([Detection(bbox=[0, 0, 10, 10], label="car", score=0.8).model_dump()]))
    [d] = ens.load_cached("sd1")
    assert d.label == "car" and d.bbox == [0, 0, 10, 10]


def test_unlabeled_tables_have_no_annotations(tmp_path):
    # Dữ liệu định dạng nuScenes chưa gán nhãn: không có sample_annotation.json -> vẫn đọc được, GT rỗng
    write_tables(tmp_path)
    data = NuScenesMini(tmp_path)
    assert data.sample_annotation == {}
    assert data._annotations_of_sample == {}
    assert [k[0] for k in data.keyframes()] == ["scene-A", "scene-A", "scene-B"]
