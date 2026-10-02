"""Bốn cải tiến 10/2026 (eval/results/improve.md): lưới ô cho detector, không hạ điểm vật ngoài tầm LiDAR, giữ box thấy
mờ (test_checks.py), mang nhãn frame trước bằng tracker."""

import numpy as np

from src.models.schemas import Detection, FrameRecord, ImageInfo, LabelObject, SweepInfo
from src.services.carry import carry_from_prev, prev_objects
from src.services.lidar2d import merge_detections


def test_merge_keeps_camera_score_when_box_has_no_lidar_points():
    far = Detection(bbox=[710, 447, 738, 471], label="car", score=0.58)
    near = Detection(bbox=[300, 400, 500, 560], label="car", score=0.9)
    uv = np.array([[400.0, 480.0], [410.0, 490.0], [420.0, 500.0]])  # chỉ xe gần có điểm LiDAR
    old = {d.bbox[0]: d.score for d in merge_detections([far, near], [], 0.4, 0.5, uv=uv)}
    assert old[710] == 0.29 and old[300] == 0.45
    new = {d.bbox[0]: d.score for d in merge_detections([far, near], [], 0.4, 0.5, uv=uv, no_lidar_scale=1.0)}
    assert new[710] == 0.58 and new[300] == 0.45  # xe gần có LiDAR mà không có box 3D: vẫn bị hạ


def test_tiles_offset_boxes_and_drop_cut_ones():
    from src.services.detectors.yoloe import YoloeDetector

    class Fake(YoloeDetector):
        def __init__(self):  # không nạp model
            from src.models.qa_config import load_autolabel_config

            self.config = load_autolabel_config("configs/autolabel.yaml")
            self.cfg = self.config.detection.yoloe.model_copy(update={"tiles": 2})

        def _predict(self, sources):
            out = []
            for im in sources:
                h, w = im.shape[:2]
                out.append([
                    Detection(bbox=[10, 10, 30, 30], label="car", score=0.5),  # bên trong ô
                    Detection(bbox=[w - 20, 10, w - 1, 30], label="car", score=0.5),  # chạm mép phải ô
                ])
            return out

    im = np.zeros((900, 1600, 3), np.uint8)
    found = Fake()._detect_tiles(im)
    boxes = sorted(d.bbox for d in found)
    # 4 ô: ô trái trên bắt đầu (0, 0); ô phải trên bắt đầu (800 - 120, 0) = (680, 0); hàng dưới y bắt đầu 450 - 67 = 383
    assert [10.0, 10.0, 30.0, 30.0] in boxes and [690.0, 10.0, 710.0, 30.0] in boxes
    # box chạm mép phải của ô trái bị bỏ (ô bị cắt); box chạm mép phải của ô phải = mép ảnh thì giữ
    assert all(not (b[2] > 900 and b[2] < 1000) for b in boxes)
    assert any(b[2] == 1599.0 for b in boxes)
    assert len(found) == 4 + 2  # 4 box trong, 2 box ở mép ảnh (hai ô bên phải)


def _frame(fid, index, ts, objects, path):
    return FrameRecord(frame_id=fid, sample_token=fid, scene="vid", index=index, camera="CAM", status="auto",
                       image=ImageInfo(sd_token=fid, path=path, timestamp=ts, width=2560, height=1440),
                       intrinsic=[[1600, 0, 800], [0, 1600, 450], [0, 0, 1]], objects=objects)  # fmt: skip


def test_prev_objects_skips_deleted_and_low_score():
    objs = [
        LabelObject(object_id="1", bbox=[0, 0, 10, 10], label="car", score=0.9),
        LabelObject(object_id="2", bbox=[0, 0, 10, 10], label="car", score=0.2),
        LabelObject(object_id="3", bbox=[0, 0, 10, 10], label="car", score=0.9),
    ]
    objs[2].review.status = "deleted"
    assert [o[2] for o in prev_objects(_frame("a", 0, 0, objs, "a.jpg"), 0.3)] == ["1"]


def test_carry_promotes_weak_detection_tracked_from_previous_keyframe(config, tmp_path):
    cfg = config.model_copy(deep=True)
    cfg.qa.temporal.carry_prev = True
    # Frame trước: xe đạp 0.34 ở x=2000, dịch dần sang phải. Frame này: detector chỉ thấy 0.20 (dưới ngưỡng giữ 0.30).
    # Không có file ảnh nên không có optical flow: tracker dự đoán theo vận tốc, vật phải dịch ít để test không phụ thuộc flow
    prev = _frame("vid_012", 12, 0, [LabelObject(object_id="3", bbox=[2000, 420, 2050, 480], label="bicycle", score=0.34)],
                  "012.jpg")  # fmt: skip
    prev.sweeps = [SweepInfo(offset=1, sd_token="s1", path="s1.jpg", timestamp=100_000,
                             detections=[Detection(bbox=[2010, 420, 2060, 480], label="bicycle", score=0.31)])]  # fmt: skip
    image = ImageInfo(sd_token="vid_013", path="013.jpg", timestamp=500_000, width=2560, height=1440)
    sweeps = {-1: ImageInfo(sd_token="s4", path="s4.jpg", timestamp=400_000, width=2560, height=1440)}
    key_dets = [Detection(bbox=[2030, 420, 2080, 480], label="bicycle", score=0.2),
                Detection(bbox=[100, 100, 200, 200], label="car", score=0.9)]  # fmt: skip
    sweep_dets = {-1: [Detection(bbox=[2020, 420, 2070, 480], label="bicycle", score=0.25)]}
    out = carry_from_prev(prev, image, sweeps, key_dets, sweep_dets, lambda p: tmp_path / p, cfg)  # không có ảnh: không flow
    assert len(out) == 1 and out[0].bbox == [2030, 420, 2080, 480] and out[0].score == 0.2
    assert out[0].source == "track" and out[0].carried_from.startswith("vid_012#3")
    # Detector đã giữ box (score cao) thì không đề xuất gì
    key_dets[0] = Detection(bbox=[2030, 420, 2080, 480], label="bicycle", score=0.6)
    assert carry_from_prev(prev, image, sweeps, key_dets, sweep_dets, lambda p: tmp_path / p, cfg) == []
