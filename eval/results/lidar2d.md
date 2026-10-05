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

## 4. Ngưỡng ghép box 2D với hình chiếu box 3D (`match_iou`) — thêm 2026-10-01

Vật nhỏ (cọc tiêu, người) có hai box lệch nhau vài px: box 3D chiếu và box YOLOE. IoU < 0.5 nên không ghép, mỗi vật
thành hai box (trên BEV box YOLOE còn bị ước lượng vị trí sai vài m). "Cặp trùng" = box 3D và box chỉ camera cùng lớp,
IoU >= 0.2, cùng qua ngưỡng 0.30.

| match_iou | dev mAP50 / F1 / cặp trùng | test mAP50 / F1 / P / cặp trùng |
|---|---|---|
| 0.5 (cũ) | 0.704 / 0.707 / 76 | 0.625 / 0.708 / 0.687 / 662 |
| **0.4 (mới)** | 0.700 / 0.708 / 32 | 0.628 / 0.722 / 0.710 / 242 |
| 0.3 | 0.687 / 0.701 / 12 | 0.626 / 0.723 / 0.687 / 60 |

Chọn 0.4 trên dev (F1 giữ nguyên, trùng -58%); 0.3 ghép nhầm cọc tiêu sang cọc bên cạnh (AP cọc tiêu dev 0.68 -> 0.60).
Ghép theo IoU giảm dần thay vì theo điểm 3D: kém hơn (test 0.620). Khớp rồi giữ box 2D cho lớp nhỏ: kém hẳn (người 0.58 ->
0.46 trên dev) vì nhãn gốc là hộp bao của box 3D.

## Đo lại với detector fine-tune toàn mạng (2026-10-04)

`python tools2d/eval_lidar2d.py --dataroot ..\v1.0-trainval` với `yoloe-26l-nuimages-full-1280.pt`, CPU từ cache. Dev 119
keyframe, test 957 keyframe (6733 vật), ngưỡng giữ 0.30. Số liệu: `eval/results/qa/lidar2d_full.json`.

| Cấu hình | dev mAP50 | test mAP50 | test P | test R | test F1 | Box thừa | Vật sót | Phải sửa |
|---|---|---|---|---|---|---|---|---|
| Chỉ detector ảnh | 0.616 | 0.589 | 0.487 | 0.807 | 0.607 | 5722 | 1301 | 7023 |
| **Gộp box 3D (mặc định, box chỉ camera x0.5)** | 0.775 | **0.697** | 0.615 | 0.821 | 0.703 | 3467 | 1206 | **4673 (−33%)** |

Tỉ lệ sai theo nguồn của box (test, trước khi hạ điểm, score ≥ 0.30):

| Nhóm | Số box | Sai |
|---|---|---|
| Cả detector ảnh và box 3D | 5196 | 15% |
| Chỉ box 3D | 680 | 67% |
| Chỉ detector ảnh, có điểm LiDAR trong box | 3209 | 68% |
| Chỉ detector ảnh, không có điểm LiDAR nào | 2744 | 95% |

Mô phỏng từ cùng dự đoán (khớp lần chạy thật ở cấu hình mặc định: test mAP50 0.695, phải sửa 4673), dò hệ số hạ điểm:

| Hệ số cho box chỉ camera | dev mAP50 | dev phải sửa | test mAP50 | test P | test R | test F1 | test phải sửa |
|---|---|---|---|---|---|---|---|
| x1.0 | 0.752 | 1258 | 0.677 | 0.490 | 0.861 | 0.625 | 6964 |
| x0.7 | 0.774 | 1114 | 0.696 | 0.537 | 0.850 | 0.658 | 5952 |
| x0.5 (mặc định) | 0.775 | 886 | 0.695 | 0.615 | 0.821 | 0.703 | 4673 |
| x0.4 | 0.772 | 657 | 0.691 | 0.706 | 0.775 | 0.739 | 3691 |
| x0.3 | 0.764 | 614 | 0.680 | 0.790 | 0.689 | 0.736 | 3327 |
| **x0.5, riêng box 0 điểm LiDAR x0.3** | **0.779** | 626 | **0.698** | 0.713 | 0.804 | **0.755** | 3504 |

Dòng cuối tốt nhất trên dev về mAP50 và gần tốt nhất về số box phải sửa; cấu hình tương ứng là
`detection.lidar3d.no_lidar_scale: 0.3`, `no_lidar_max_points: 0`. Đã áp vào config và chạy bằng code thật trên 6 camera
(mục dưới).

Dùng box 3D chỉ làm ý kiến thứ hai cho QA (không đổi nhãn): AUC xếp box sai trên test 0.913 → 0.918, nhóm medium phải xem
kỹ 4115 → 3597 box. Lợi ích nhỏ, vì số điểm LiDAR trong box đã cho gần hết thông tin đó. Lợi ích lớn đến từ việc gộp box
3D vào nhãn (bảng đầu), không phải từ QA.

Giới hạn như mục 1: nhãn gốc 2D là box 3D chiếu xuống nên một phần mức tăng do cùng kiểu hộp; chỉ áp dụng cho dự án có
LiDAR và đã chạy 4 mô hình 3D.

## Cả 6 camera, cấu hình mới (2026-10-04)

`python tools2d/eval_lidar2d.py --dataroot ..\v1.0-trainval --cameras all`, chạy bằng code thật (detector trên GPU cho 5
camera chưa có cache, phần gộp trên CPU). 1076 keyframe × 6 camera: dev 714 ảnh (4594 vật), test 5742 ảnh (25991 vật).
Số liệu: `eval/results/qa/lidar2d_6cam.json`.

Test, cả 6 camera:

| Cấu hình | mAP50 | P | R | F1 | Box thừa | Vật sót | Phải sửa |
|---|---|---|---|---|---|---|---|
| Chỉ detector ảnh | 0.560 | 0.534 | 0.763 | 0.628 | 17262 | 6170 | 23432 |
| Gộp box 3D, mọi box chỉ camera x0.5 (trước 04/10) | 0.671 | 0.628 | 0.818 | 0.710 | 12612 | 4736 | 17348 |
| **Gộp box 3D, box 0 điểm LiDAR x0.3 (mặc định mới)** | **0.672** | **0.672** | 0.812 | **0.735** | 10287 | 4897 | **15184 (−35%)** |

Dev: mAP50 0.564 → 0.678 → 0.680; phải sửa 4063 → 3555 → 3176.

Từng camera trên test (chỉ detector ảnh → mặc định mới):

| Camera | Số vật | mAP50 | P | R | F1 | Phải sửa |
|---|---|---|---|---|---|---|
| CAM_FRONT | 6733 | 0.589 → 0.700 | 0.487 → 0.713 | 0.807 → 0.804 | 0.607 → 0.755 | 7023 → 3504 |
| CAM_FRONT_LEFT | 2936 | 0.602 → 0.741 | 0.545 → 0.692 | 0.765 → 0.839 | 0.637 → 0.758 | 2562 → 1572 |
| CAM_FRONT_RIGHT | 3456 | 0.553 → 0.662 | 0.522 → 0.622 | 0.759 → 0.839 | 0.618 → 0.714 | 3238 → 2322 |
| CAM_BACK | 7191 | 0.576 → 0.685 | 0.603 → 0.702 | 0.735 → 0.788 | 0.663 → 0.742 | 5384 → 3932 |
| CAM_BACK_LEFT | 2338 | 0.557 → 0.684 | 0.560 → 0.694 | 0.756 → 0.837 | 0.643 → 0.759 | 1958 → 1244 |
| CAM_BACK_RIGHT | 3337 | 0.579 → 0.658 | 0.507 → 0.578 | 0.739 → 0.808 | 0.602 → 0.674 | 3268 → 2611 |

- Gộp box 3D có lợi ở cả 6 camera: mAP50 +0.08 đến +0.14, số box phải sửa −20% đến −50%.
- Hệ số riêng x0.3 cho box 0 điểm LiDAR giúp nhiều nhất ở CAM_FRONT (phải sửa 4673 → 3504); cả 6 camera: 17348 →
  15184 (−12%), precision 0.628 → 0.672, recall 0.818 → 0.812, mAP50 gần như không đổi.
- CAM_BACK_RIGHT yếu nhất sau khi gộp (P 0.578, F1 0.674); chưa tìm nguyên nhân.
- Số của CAM_FRONT khớp phần mô phỏng ở mục trên (mAP50 0.700 so với 0.698, phải sửa 3504).

## Toạ độ của box trùng cả hai nguồn: `box_blend` (2026-10-04)

Box 3D chiếu xuống ảnh là hộp bao 8 đỉnh nên rộng hơn vật: box gộp có cả hai nguồn rộng 1.31 lần box của detector ảnh.
`detection.lidar3d.box_blend`: toạ độ = `box_blend` x box detector + (1 − `box_blend`) x box 3D chiếu. Chấm trên
CAM_FRONT test 957 keyframe (nhãn gốc nuScenes là hình chiếu box 3D nên ưu ái box rộng), ngưỡng giữ 0.30:

| Toạ độ | mAP50 | P | R | F1 | F1 @IoU 0.75 | Diện tích / box detector |
|---|---|---|---|---|---|---|
| Box 3D chiếu (0, trước đây) | 0.677 | 0.490 | 0.861 | 0.625 | 0.367 | 1.31 |
| Box detector (1) | 0.663 | 0.483 | 0.851 | 0.616 | 0.278 | 1.00 |
| **Trung bình (0.5, mặc định)** | **0.687** | 0.493 | 0.866 | **0.628** | **0.394** | 1.15 |
| Giao hai box | 0.657 | 0.480 | 0.846 | 0.612 | 0.242 | 0.98 |

Dev cùng chiều (mAP50 0.760 > 0.752 > 0.741). Mới đo trên CAM_FRONT; chạy lại cả 6 camera:
`python tools2d/eval_lidar2d.py --dataroot ..\\v1.0-trainval --cameras all --all-variants`.

## Giữ box detector thấy rõ dù không có điểm LiDAR: `keep_det_score` (2026-10-05)

Lỗi trên giao diện: scene-0097 keyframe 34 chỉ còn 6 box trong khi các sweep quanh nó có 14 detection; xe đỗ bên trái
detector cho 0.91 nhưng không có điểm LiDAR nào trong box nên điểm bị hạ còn 0.27 (x0.3), dưới ngưỡng giữ 0.30, và bị
bỏ. Keyframe trước và sau có box 3D trùng xe đó nên nhãn hiện lại: box nhấp nháy giữa các keyframe.

`detection.lidar3d.keep_det_score: 0.8`: box chỉ camera thấy mà điểm gốc của detector >= 0.8 thì vẫn giữ. Điểm đã hạ
giữ nguyên, nên box vào nhóm rủi ro cao (LOW_CONFIDENCE + NO_LIDAR_SUPPORT) chứ không vào nhóm duyệt theo lô.
Keyframe 34 của scene-0097: 6 → 10 box (thêm 4 xe 0.91 / 0.89 / 0.88 / 0.88).

Chấm theo nhãn gốc nuScenes trên bản dump CAM_FRONT (`data/lidar2d_dump.json`, ghép IoU 0.5 cùng lớp, ngưỡng giữ 0.30;
bộ ghép đơn giản hơn `eval2d.evaluate` nên số gốc lệch nhẹ so với bảng trên):

| Luật giữ box | Tập | Số box | P | R | F1 | Box thừa | Vật sót | Phải sửa |
|---|---|---|---|---|---|---|---|---|
| Điểm đã hạ >= 0.30 (trước đây) | dev | 1423 | 0.668 | 0.821 | 0.737 | 438 | 193 | 631 |
| + điểm gốc >= 0.90 | dev | 1424 | 0.668 | 0.821 | 0.736 | 439 | 193 | 632 |
| **+ điểm gốc >= 0.80 (mặc định mới)** | dev | 1536 | 0.633 | 0.840 | 0.722 | 523 | 172 | 695 |
| + điểm gốc >= 0.70 | dev | 1636 | 0.597 | 0.843 | 0.699 | 613 | 169 | 782 |
| + điểm gốc >= 0.60 | dev | 1750 | 0.557 | 0.844 | 0.671 | 723 | 168 | 891 |
| Điểm đã hạ >= 0.30 (trước đây) | test | 8753 | 0.707 | 0.800 | 0.751 | 2235 | 1346 | 3581 |
| + điểm gốc >= 0.90 | test | 8755 | 0.707 | 0.800 | 0.751 | 2236 | 1345 | 3581 |
| **+ điểm gốc >= 0.80 (mặc định mới)** | test | 9317 | 0.673 | 0.813 | 0.736 | 2664 | 1261 | 3925 |
| + điểm gốc >= 0.70 | test | 9764 | 0.642 | 0.816 | 0.719 | 3059 | 1238 | 4297 |
| + điểm gốc >= 0.60 | test | 10263 | 0.610 | 0.817 | 0.698 | 3521 | 1231 | 4752 |

- Theo nhãn gốc nuScenes thì luật mới **kém hơn**: test thêm 564 box, trong đó 85 trùng nhãn gốc; precision 0.707 →
  0.673, recall 0.800 → 0.813, phải sửa 3581 → 3925 (+10%).
- Con số này không đo đúng thứ người dùng thấy: nuScenes chỉ gán nhãn cho vật có ít nhất một điểm LiDAR / radar, nên
  xe thấy rõ trên ảnh mà không có điểm LiDAR không có nhãn gốc và bị tính là box thừa. Chưa có bộ nhãn 2D vẽ theo
  ảnh để chấm công bằng; tỉ lệ box loại này là vật thật chưa được đo.
- 0.90 gần như không giữ thêm gì (detector hiếm khi cho điểm trên 0.9); dưới 0.80 số box thừa tăng nhanh hơn số vật
  tìm lại. Chọn 0.80. Tắt: `keep_det_score: null`.
- Chỉ đo CAM_FRONT. Luật không đụng tới box có box 3D, nên mAP của phần gộp 3D không đổi.
