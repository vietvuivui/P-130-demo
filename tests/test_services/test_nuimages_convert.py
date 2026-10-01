"""tools2d/nuimages_to_yolo.py: nhãn YOLO đúng lớp / toạ độ, bỏ box quá nhỏ, loại ảnh trùng log với scene val nuScenes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def conv():
    sys.path.insert(0, str(ROOT / "tools2d"))
    spec = importlib.util.spec_from_file_location("nuimages_to_yolo", ROOT / "tools2d" / "nuimages_to_yolo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_nuimages(root: Path, version: str, logs: list[tuple[str, str]]):
    (root / version).mkdir(parents=True)
    (root / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
    cats = [{"token": "c1", "name": "vehicle.car"}, {"token": "c2", "name": "movable_object.trafficcone"},
            {"token": "c3", "name": "animal"}]  # fmt: skip
    tabs = {"category": cats, "log": [], "sample": [], "sample_data": [], "object_ann": []}
    for i, (vehicle, date) in enumerate(logs):
        fn = f"samples/CAM_FRONT/{version}_{i}.jpg"
        Image.new("RGB", (1600, 900)).save(root / fn)
        tabs["log"].append({"token": f"l{i}", "vehicle": vehicle, "date_captured": date, "logfile": "", "location": ""})
        tabs["sample"].append({"token": f"s{i}", "log_token": f"l{i}"})
        tabs["sample_data"] += [
            {"token": f"sd{i}", "sample_token": f"s{i}", "filename": fn, "width": 1600, "height": 900, "is_key_frame": True},
            {"token": f"sw{i}", "sample_token": f"s{i}", "filename": "sweeps/x.jpg", "width": 1600, "height": 900, "is_key_frame": False},
        ]  # fmt: skip
        tabs["object_ann"] += [
            {"token": f"a{i}", "sample_data_token": f"sd{i}", "category_token": "c1", "bbox": [400, 300, 800, 600]},
            {"token": f"b{i}", "sample_data_token": f"sd{i}", "category_token": "c2", "bbox": [10, 10, 12, 30]},  # 2 px
            {"token": f"d{i}", "sample_data_token": f"sd{i}", "category_token": "c3", "bbox": [0, 0, 100, 100]},  # lớp ngoài
        ]  # fmt: skip
    for name, rows in tabs.items():
        (root / version / f"{name}.json").write_text(json.dumps(rows))


def test_convert_split_labels_and_leak_guard(conv, tmp_path):
    src = tmp_path / "nuim"
    fake_nuimages(src, "v1.0-train", [("n015", "2018-07-24"), ("n008", "2018-08-01")])
    classes, cat_map = conv.class_names(ROOT / "configs" / "autolabel.yaml")
    out = tmp_path / "yolo"
    stats = conv.convert_split(
        src, "v1.0-train", "train", out, classes, cat_map, {("n008", "2018-08-01")}, "copy", None
    )
    assert stats["images"] == 1 and stats["dropped_same_log_as_nuscenes_val"] == 1
    assert stats["boxes"] == {"car": 1}  # cone 2 px và lớp ngoài taxonomy bị bỏ
    lines = (out / "labels" / "train" / "v1.0-train_0.txt").read_text().split()
    assert lines == [str(classes.index("car")), "0.375000", "0.500000", "0.250000", "0.333333"]
    assert (out / "images" / "train" / "v1.0-train_0.jpg").exists()
    assert not (out / "labels" / "train" / "v1.0-train_1.txt").exists()
