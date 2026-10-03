"""Bấm-để-chọn-vật: mask -> box + đa giác, GrabCut dự phòng trên ảnh tổng hợp, SAM giả, và API /frames/{id}/segment."""

import numpy as np
import pytest

from src.services import segment


def _scene(path):
    cv2 = pytest.importorskip("cv2")
    img = np.full((300, 400, 3), 120, np.uint8)
    img[::7] = 110  # nền có vân nhẹ
    cv2.rectangle(img, (150, 100), (260, 180), (30, 30, 210), -1)  # "xe" đỏ
    cv2.imwrite(str(path), img)
    return path


def test_mask_to_result_keeps_largest_blob():
    m = np.zeros((100, 120), np.uint8)
    m[20:60, 30:90] = 1
    m[80:84, 5:9] = 1  # mảnh vụn
    r = segment.mask_to_result(m, 0.9, "sam2")
    assert r["bbox"] == [30.0, 20.0, 90.0, 60.0] and r["engine"] == "sam2" and len(r["polygon"]) >= 8
    with pytest.raises(segment.SegmentError):
        segment.mask_to_result(np.zeros((10, 10), np.uint8), 0.1, "sam2")


def test_grabcut_fallback_finds_the_clicked_object(tmp_path):
    p = _scene(tmp_path / "a.jpg")
    r = segment.segment_points(p, [[205, 140]], engine="grabcut")
    x1, y1, x2, y2 = r["bbox"]
    assert r["engine"] == "grabcut"
    # GrabCut chỉ là dự phòng CPU (dựa vào màu): yêu cầu trúng vật với IoU >= 0.7, không đòi sát từng pixel như SAM
    ix, iy = min(x2, 261) - max(x1, 150), min(y2, 181) - max(y1, 100)
    union = (x2 - x1) * (y2 - y1) + 111 * 81 - ix * iy
    assert ix > 0 and iy > 0 and ix * iy / union >= 0.7
    with pytest.raises(segment.SegmentError):
        segment.segment_points(p, [], engine="grabcut")
    with pytest.raises(segment.SegmentError):
        segment.segment_points(tmp_path / "nope.jpg", [[1, 1]])


def test_sam_engine_is_used_when_available(tmp_path, monkeypatch):
    p = _scene(tmp_path / "a.jpg")
    calls = []

    def fake_sam(path, points, labels):
        calls.append((points, labels))
        m = np.zeros((300, 400), np.uint8)
        m[100:181, 150:261] = 1
        return segment.mask_to_result(m, 0.97, "sam2")

    monkeypatch.setattr(segment, "onnx_available", lambda: "chưa có mô hình")
    monkeypatch.setattr(segment, "sam_available", lambda: None)
    monkeypatch.setattr(segment, "_segment_sam", fake_sam)
    r = segment.segment_points(p, [[205, 140], [300, 250]], [1, 0])
    assert r["engine"] == "sam2" and r["bbox"] == [150.0, 100.0, 261.0, 181.0]
    assert calls == [([[205, 140], [300, 250]], [1, 0])]

    def broken(path, points, labels):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(segment, "_segment_sam", broken)
    monkeypatch.setitem(segment._SAM, "failed", None)
    r = segment.segment_points(p, [[205, 140]])  # SAM lỗi -> tự chuyển GrabCut, lần sau khỏi thử lại
    assert r["engine"] == "grabcut" and "CUDA" in segment._SAM["failed"]
    monkeypatch.setitem(segment._SAM, "failed", None)


def test_pick_mask_prefers_whole_object_for_a_single_click():
    masks = np.zeros((3, 8, 8), np.float32)
    masks[0, :1, :1] = 1  # mảnh nhỏ, điểm thấp
    masks[1, :3, :3] = 1  # bộ phận, điểm cao nhất
    masks[2, :6, :6] = 1  # cả vật, điểm đủ tin
    assert segment.pick_mask(masks, np.array([0.15, 0.93, 0.76]), single_point=True) == 2
    assert segment.pick_mask(masks, np.array([0.15, 0.93, 0.60]), single_point=True) == 1  # cả vật không đủ tin
    assert segment.pick_mask(masks, np.array([0.15, 0.93, 0.76]), single_point=False) == 1  # prompt rõ: điểm cao nhất
    assert segment.pick_mask(masks, np.array([0.2, 0.3, 0.1]), single_point=True) == 1  # không mask nào đủ tin


def test_box_only_prompt_falls_back_to_centre_point_without_sam(tmp_path, monkeypatch):
    p = _scene(tmp_path / "a.jpg")
    monkeypatch.setattr(segment, "onnx_available", lambda: "chưa có mô hình")
    monkeypatch.setattr(segment, "sam_available", lambda: "không có GPU")
    r = segment.segment_points(p, [], box=[140, 90, 270, 190])
    assert r["engine"] == "grabcut" and r["bbox"][0] < 205 < r["bbox"][2]
    with pytest.raises(segment.SegmentError):
        segment.segment_points(p, [])


@pytest.mark.skipif(segment.onnx_available() is not None, reason="chưa có SAM 2.1 ONNX (scripts/download_sam_onnx.py)")
def test_sam_onnx_point_and_box_prompts(tmp_path):
    """Chạy thật SAM 2.1 ONNX trên CPU: bấm vào 'xe' và kéo box thô đều cho box sát vật; embedding được dùng lại."""
    p = _scene(tmp_path / "a.jpg")
    truth = [150, 100, 261, 181]

    def iou(b):
        ix, iy = min(b[2], truth[2]) - max(b[0], truth[0]), min(b[3], truth[3]) - max(b[1], truth[1])
        return ix * iy / ((b[2] - b[0]) * (b[3] - b[1]) + 111 * 81 - ix * iy)

    r = segment.segment_points(p, [[205, 140]])
    assert r["engine"] == "sam2-onnx" and iou(r["bbox"]) > 0.85 and r["polygon"]
    n = len(segment._ONNX["cache"])
    r2 = segment.segment_points(p, [], box=[130, 85, 280, 195])
    assert iou(r2["bbox"]) > 0.85 and len(segment._ONNX["cache"]) == n  # cùng ảnh: không mã hoá lại
    r3 = segment.segment_points(p, [[205, 140], [60, 60]], [1, 0])
    assert iou(r3["bbox"]) > 0.85
