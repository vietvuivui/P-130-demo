# Box "thở" khi phát video — 2026-10-04

Khi phát video (`GET /videos/{id}/playback`), box ở sweep t±1, t±2 là detection độc lập của từng ảnh
(`src/agents/nodes/temporal.py`: `o.track[offset] = dets[j].bbox`), sweep không có detection thì vẽ nguyên box keyframe.
YOLO ra w/h lệch vài % mỗi ảnh nên box co giãn liên tục.

Thước đo: giật = sai phân bậc hai của w, h trên 3 ảnh liên tiếp của cùng object (phóng to đều khi vật tới gần không tính).
Đo trên chuỗi playback thật của repo (script ngoài repo, CPU).

| Video | cặp | giật > 5% trước → sau | > 10% trước → sau |
|---|---|---|---|
| dashcam 2560×1440 (`p-133699-…`, 62 keyframe) | 178 | 43.3% → **2.8%** | 23.0% → 2.2% |
| dashcam trong `p-nusc3d-scene-0101` | 178 | 41.0% → **3.9%** | 21.9% → 3.4% |

Cách làm (`src/services/video.py: smooth_track_boxes`): tâm và kích thước của sweep khớp đường thẳng theo thời gian **ép đi
qua box keyframe** (box keyframe không đổi → nhãn, QA, xuất file giữ nguyên, chỉ hiển thị khác). Sweep có detection: tâm
theo detection, kích thước trên đường thẳng chặn ±20% so với detection đó; sweep thiếu detection: cả tâm lẫn kích thước trên
đường thẳng (thay cho vẽ box keyframe sai chỗ).

Các biến thể đã thử trên sweep (mô phỏng, dashcam 144 cặp / nuScenes held-out 795 keyframe 22811 cặp; IoU = so với
detection gốc cùng ảnh):

| Biến thể | giật >5% dashcam | nuScenes | IoU trung vị / tệ nhất (nuScenes) |
|---|---|---|---|
| raw | 38.2% | 59.1% | 1 / 1 |
| đường thẳng tự do (đổi cả keyframe) | 0.7% | 1.2% | 0.971 / 0.315 |
| qua keyframe, không chặn | 0.7% | 1.3% | 0.968 / 0.129 |
| **qua keyframe, chặn ±20% (chọn)** | 3.5% | 5.9% | 0.968 / 0.640 |
| EMA α=0.5 | 18.8% | 41.3% | 0.949 / 0.448 |
| trung vị 5 ảnh | 0% | 0% | 0.927 / 0.312 (ép vật đang tới gần về cỡ keyframe) |

Đã thử và **không làm**: `Track.output_box()` trả box đã làm mượt khi lan truyền giữa keyframe — ws_dev_full 1197 box lan
truyền, IoU TB 0.637 → 0.631 (t.box) / 0.635 (cỡ t.box, tâm detection), số đúng không đổi; lan truyền chỉ ghi box ở
keyframe nên không giúp phần phát video.

Ảnh minh hoạ: `playback_before_after.jpg` (đỏ: trước, xanh: sau; xe tải `vid-…_006` object 2, w cũ 339/321/292/248/256,
mới 340/316/292/268/244).
