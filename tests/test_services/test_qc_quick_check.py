import json

import pytest

from src.models.schemas import ReviewActionRequest
from src.services import qc, review
from src.services.exporter import export_dataset
from src.services.qc.quick_check import parse_labels

FID = "scene-0001_000"


def run(store, config, payload, name="labels.json"):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return qc.quick_check(store, config, name, data)


def findings(res):
    return [f for fr in res.frames for f in fr.findings]


def test_parse_coco_jsonl_and_frames(config):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 1600, "height": 900}],
        "annotations": [{"id": 5, "image_id": 1, "category_id": 2, "bbox": [10, 20, 30, 40]}],
        "categories": [{"id": 2, "name": "vehicle.car"}],
    }
    fmt, [f], errors = parse_labels("x.json", json.dumps(coco).encode(), config)
    assert fmt == "coco" and not errors
    assert (f.labels[0].label, f.labels[0].bbox) == ("car", [10, 20, 40, 60])  # category nuScenes -> car, xywh -> xyxy

    jsonl = b'{"frame_id": "f1", "objects": [{"bbox": [1, 2, 30, 40], "class": "person"}]}\nnot json\n'
    fmt, [f], errors = parse_labels("x.jsonl", jsonl, config)
    assert fmt == "jsonl" and f.labels[0].label == "pedestrian" and errors == ["Dòng 2: JSON hỏng (Expecting value)"]

    fmt, frames, _ = parse_labels(
        "x.json", json.dumps({"frames": [{"frame_id": "f1", "objects": []}]}).encode(), config
    )
    assert fmt == "frames" and frames[0].frame_id == "f1"
    with pytest.raises(qc.QCError):
        parse_labels("x.json", b'{"foo": 1}', config)


def test_frame_outside_workspace_gets_structural_checks(store, config):
    res = run(
        store,
        config,
        {
            "frames": [
                {
                    "frame_id": "elsewhere_001",
                    "width": 1600,
                    "height": 900,
                    "objects": [
                        {"bbox": [0, 0, "x", 5], "class": "car"},
                        {"bbox": [10, 10, 60, 60], "class": "unicorn"},
                        {"bbox": [1500, 10, 1700, 60], "class": "car"},
                    ],
                }
            ]
        },
    )
    assert res.n_frames == 1 and res.n_labels == 2 and not res.frames[0].in_workspace
    assert sorted(f.code for f in findings(res)) == [
        "BOX_OUT_OF_IMAGE",
        "FRAME_NOT_IN_WORKSPACE",
        "INVALID_BOX",
        "UNKNOWN_CLASS",
    ]
    assert res.n_errors == 3 and res.n_warnings == 1


def test_missing_and_disagreeing_labels_against_workspace(store, config):
    # Workspace có detection: #1 car 0.92, #2 pedestrian 0.80, #3 barrier 0.39.
    # File thiếu #1 và gán #2 (box dáng người) thành car
    res = run(
        store, config, {"frames": [{"frame_id": FID, "objects": [{"bbox": [700, 420, 760, 560], "class": "car"}]}]}
    )
    by_code = {f.code: f for f in findings(res)}
    # Hai tín hiệu độc lập cùng bắt nhãn sai lớp: lớp khác model + tỉ lệ box không hợp với car.
    # Barrier score 0.39 < 0.5 nên không bị gợi ý là vật bị sót
    assert set(by_code) == {"POSSIBLY_MISSING", "MODEL_DISAGREES", "ASPECT_RATIO_ABNORMAL"}
    assert by_code["POSSIBLY_MISSING"].suggestion == {"action": "ADD_BOX", "bbox": [100, 400, 300, 520], "label": "car"}
    assert by_code["MODEL_DISAGREES"].suggestion["label"] == "pedestrian"
    assert res.frames[0].in_workspace and res.elapsed_ms >= 0

    # Người đã xoá #1 trong workspace (FP đã xác nhận) -> không gợi ý thêm lại
    f = store.load_frame(FID)
    review.apply_action(f, ReviewActionRequest(action="DELETE", object_id="1"), "an", config)
    store.save_frame(f)
    res = run(
        store,
        config,
        {"frames": [{"frame_id": FID, "objects": [{"bbox": [700, 420, 760, 560], "class": "pedestrian"}]}]},
    )
    assert findings(res) == []


def test_export_passes_quick_check(store, config):
    """Round-trip: file export của chính hệ thống phải qua Quick Check sạch."""
    f = store.load_frame(FID)
    review.approve_low_risk(f, "an")
    review.apply_action(f, ReviewActionRequest(action="KEEP", object_id="3"), "an", config)
    review.approve_frame(f, "an", 5)
    store.save_frame(f)
    out = store.exports_dir / export_dataset(store, config).export_id

    for name in ("coco.json", "labels.jsonl"):
        res = run(store, config, (out / name).read_bytes(), name=name)
        assert res.n_frames == 1 and res.n_labels == 3 and res.frames[0].in_workspace, name
        assert findings(res) == [], name


def test_decisions_recorded_in_file_are_respected(store, config):
    # Box 100x30 px gán traffic_cone: tỉ lệ sai với lớp -> ASPECT_RATIO_ABNORMAL mỗi lần kiểm lại
    box = [900, 300, 1000, 330]
    objs = [
        # Người thấy issue mà vẫn KEEP -> đã xem, không báo lại
        {
            "object_id": "a",
            "bbox": box,
            "class": "traffic_cone",
            "human_action": "KEEP",
            "issues": ["ASPECT_RATIO_ABNORMAL"],
        },
        # Người đã xác nhận cảnh báo QC
        {
            "object_id": "b",
            "bbox": [1100, 300, 1200, 330],
            "class": "traffic_cone",
            "human_action": "ADD_BOX",
            "qc_acknowledged": ["ASPECT_RATIO_ABNORMAL"],
        },
        # Issue ghi cho box cũ; người đã sửa box -> phải kiểm lại
        {
            "object_id": "c",
            "bbox": [1300, 300, 1400, 330],
            "class": "traffic_cone",
            "human_action": "EDIT_BOX",
            "issues": ["ASPECT_RATIO_ABNORMAL"],
        },
    ]
    res = run(store, config, {"frames": [{"frame_id": "elsewhere", "width": 1600, "height": 900, "objects": objs}]})
    aspect = {f.object_id: f.acked for f in findings(res) if f.code == "ASPECT_RATIO_ABNORMAL"}
    assert aspect == {"a": True, "b": True, "c": False}
    assert res.n_acked == 2 and res.by_code == {"ASPECT_RATIO_ABNORMAL": 1, "FRAME_NOT_IN_WORKSPACE": 1}
