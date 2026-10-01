# Box 3D chiếu xuống ảnh làm nhãn 2D, và ngưỡng nối track 3D (2026-10-01)

Chọn trên **dev** (3 scene demo, 119 keyframe), báo trên **test** (24 scene val khác, 957 keyframe CAM_FRONT). So với nhãn
gốc nuScenes. Chạy trên CPU từ dự đoán đã cache (`tools3d/work/preds`, `data/cache_eval2d`); chưa chạy qua web với GPU.

## 1. Gộp box 3D vào nhãn 2D (`src/services/lidar2d.py`, bước dự án `fuse2d`)

Box 3D của ensemble chiếu xuống ảnh. Box YOLOE trùng (IoU >= 0.5, cùng lớp): giữ box 3D chiếu, điểm = max. Box YOLOE không
trùng: điểm x `camera_only_scale`. Ngưỡng giữ box 0.30 như sản phẩm.

| Cấu hình | dev mAP50 | test mAP50 | test P | test R | test F1 |
|---|---|---|---|---|---|
| YOLOE fine-tune (trước) | 0.434 | 0.366 | 0.467 | 0.582 | 0.518 |
| Chỉ box 3D chiếu | 0.585 | 0.577 | 0.805 | 0.654 | 0.722 |
| Gộp, camera_only_scale 1.0 | 0.642 | 0.564 | 0.488 | 0.785 | 0.602 |
| Gộp, camera_only_scale 0.7 | 0.674 | 0.607 | 0.581 | 0.767 | 0.661 |
| **Gộp, camera_only_scale 0.5 (mặc định mới)** | **0.704** | **0.625** | **0.687** | **0.730** | **0.708** |

Ba dòng giữa đo với ensemble cũ (track_min_score 0.1); dòng cuối là đúng code trong repo (ensemble mới).

Số box người phải sửa trên test (box thừa phải xoá + vật sót phải vẽ, ở ngưỡng 0.30, 6733 vật):

| | Đúng | Thừa | Sót | Phải sửa | Box trên vật khuất > 60% (không tính lỗi) |
|---|---|---|---|---|---|
| YOLOE | 3917 | 4475 | 2816 | 7291 | 474 |
| Gộp | 4912 | 2239 | 1821 | **4060 (-44%)** | 967 |

Dev: phải sửa 1068 -> 655 (-39%).

Giới hạn:

- Nhãn gốc 2D của nuScenes cũng là box 3D chiếu xuống, nên một phần mức tăng là do cùng kiểu hộp (bao cả phần bị che).
- Mô hình LiDAR thấy cả vật bị che trong ảnh: số box nằm trên vật khuất > 60% tăng gấp đôi. Bộ chấm bỏ qua các box này;
  người duyệt có thể muốn xoá.
- Chưa đo qua QA Agent (mức risk của box đến từ 3D) và chưa chạy `simulate_review.py`: cần GPU và server.
- Mới đo CAM_FRONT.

## 2. `track_min_score` 0.1 -> 0.05 (`tools3d/refine3d.py`)

| | dev mAP | test mAP | test NDS (không AAE) |
|---|---|---|---|
| 0.1 (trước) | 0.644 | 0.666 | 0.714 |
| **0.05** | **0.654** | **0.671** | **0.719** |
| 0.03 / 0.02 (dev) | 0.645 / 0.644 | | |

Ở ngưỡng hiển thị 0.30 của sản phẩm thì gần như không đổi: test P 0.792 -> 0.804, R 0.817 -> 0.800, F1 0.804 -> 0.802.
Lợi ích nằm ở thứ hạng box (mAP) và ở nhãn 2D gộp (test mAP50 0.618 -> 0.625).

## 3. Đã thử, không dùng

- Bỏ bớt mô hình trong ensemble (test mAP, có track): cả 4 0.666; bỏ SSN 0.664; bỏ CenterPoint pillar 0.657; bỏ
  PointPillars 0.648; bỏ CenterPoint voxel 0.632. Dev: bỏ SSN 0.615 (giảm). Giữ cả 4.
- Trọng số theo độ mạnh mô hình, bán kính gộp x0.75 / x1.5, blend 0.3 / 0.7: giảm trên dev.

Bộ chấm 3D ở đây viết lại bằng numpy (không AAE, không lọc bike rack), khớp số của `run3d.py eval` trong 0.002 mAP.
