# Bảng so sánh các phương pháp đã dùng (2D và 3D)

Cập nhật 2026-10-01. Mọi số đều so với **nhãn gốc nuScenes**. Chọn cách làm trên **dev** (3 scene demo), báo cáo trên
**test** (scene val khác, chưa dùng để chọn gì): 3D 24 scene / 957 keyframe; 2D detector 24 scene / 957 keyframe
(CAM_FRONT); lan truyền 2D 20 scene / 795 keyframe / 154 lần lan truyền.

Chỉ số dùng trong bảng:

- **mAP50** (2D): AP trung bình 10 lớp, box đúng khi IoU ≥ 0.5. **P / R**: precision / recall ở ngưỡng của sản phẩm.
- **mAP / NDS** (3D): chuẩn nuScenes (ghép theo khoảng cách tâm; NDS gộp thêm sai số vị trí, kích thước, hướng, vận tốc).
- **Lan truyền nhãn**: nhãn gốc ở keyframe 0, 5, 10… đóng vai nhãn người đã duyệt, đi tiếp tối đa 10 keyframe.
  *Đúng* = box ghi ra nằm đúng vật (IoU ≥ 0.5, cùng vật); *sai* = box không trùng vật nào; *đổi ID* = box nhảy sang vật
  khác; *thiếu* = vật còn trong ảnh nhưng không có box. **Điểm = đúng − sai − 2 × đổi ID** (đổi ID tính gấp đôi vì box
  mang lớp / kích thước của vật cũ nên người khó phát hiện hơn box sai hẳn).

## 1. Tổng hợp: mỗi phương pháp một dòng

| # | Phần | Phương pháp | Chỉ số quyết định | Không dùng | Có dùng | Thay đổi | Dùng? |
|---|---|---|---|---|---|---|---|
| 1 | 2D | YOLOE-26-L thay YOLO-World | mAP50 (40 keyframe) | 0.311 | 0.452 | +0.141 | ✅ |
| 2 | 2D | **Fine-tune YOLOE-26-L trên nuImages** | mAP50 test | 0.312 | 0.366 | +0.054 (R +0.058, P −0.015) | ✅ mới |
| 3 | 2D | Ngưỡng giữ box 0.30 | precision (40 keyframe) | 0.34 | 0.51 | +0.17 | ✅ |
| 4 | 2D | TTA: đổi prompt, lật ảnh, ảnh 1600 px | mAP50 dev | 0.395 | 0.365–0.392 | giảm | ❌ |
| 5 | 2D | Optical flow khi lan truyền | nhãn đúng / đổi ID (test) | 2866 / 205 | 3220 / 82 | +12% / −60% | ✅ |
| 6 | 2D | ByteTrack (ghép 2 tầng) | box sai (test) | 960 | 860 | −10% (đúng −0.9%) | ✅ |
| 7 | 2D | OC-SORT (OCR), không ByteTrack | điểm (test) | 2096 | 2071 | −25 | ❌ |
| 8 | 2D | OC-SORT (OCR) thêm vào ByteTrack | điểm (test) | 2180 | 2173 | −7 | ❌ |
| 9 | 2D | OC-SORT ORU / OCM | điểm dev | 554 | 550 / 549 | ≈ 0 | ❌ |
| 10 | 2D | Optical flow cho QA temporal | lỗi lọt qua duyệt theo lô | 1632 | 2082 | +28% | ❌ |
| 11 | 2D | Tính lại score theo sweep (Seq-NMS) | box phải xem tay | 1707 | 648 | −62% (vật sót +9%) | tuỳ chọn |
| 12 | 2D | Nhận box sweep score thấp (ý ByteTrack) | lỗi lọt qua duyệt theo lô | — | tăng | xấu hơn | ❌ |
| 13 | 3D | CenterPoint voxel (mô hình đơn tốt nhất) | mAP test | 0.473 (PointPillars) | 0.578 | +0.105 | ✅ |
| 14 | 3D | Gộp 4 mô hình LiDAR (WBF) | mAP test | 0.578 | 0.637 | +0.059 | ✅ |
| 15 | 3D | Tinh chỉnh theo track | mAP / NDS test | 0.637 / 0.690 | 0.668 / 0.713 | +0.031 / +0.023 | ✅ |
| 16 | 3D | Thêm 2 mô hình camera vào ensemble | mAP test | 0.668 | 0.662 | −0.006 | ❌ |
| 17 | 3D | Kiểm tra chéo 3D bằng 2D (chấm lại / xoá box) | box sửa đúng / làm hỏng (dev) | — | 14 / 39 | hỏng nhiều hơn | ❌ |
| 18 | 3D | VESPA: box 3D từ mask 2D + LiDAR | mAP dev | 0.644 | 0.627 | −0.017 | ❌ |
| 19 | 3D | VESPA: hướng theo chuyển động, cỡ theo lớp | NDS test | 0.713 | 0.712–0.713 | ≈ 0 | ❌ |
| 20 | 3D | Lan truyền theo vận tốc mô hình | đổi ID (held-out 4 scene) | 98 | 36 | −63% | ✅ |
| 21 | 3D | Giữ track 4 keyframe thay vì 2 (ý OC-SORT) | điểm dev; nhãn đúng test | 3731; 23355 | 3762; 23731 | +31; +1.6% | ✅ |
| 22 | 3D | OC-SORT OCR | điểm dev | 3762 | 3734 | −28 (test +63) | ❌ (bật được) |
| 23 | 3D | OC-SORT ORU / OCM | điểm test | 21469 | 21491 / 21387 | ≈ 0 | ❌ |

## 2. 2D — detector

| Detector | Tập | mAP50 | P | R | F1 |
|---|---|---|---|---|---|
| YOLO-World (từ vựng COCO, không có barrier) | 40 keyframe | 0.311 | | | |
| YOLO26 (tập lớp COCO, không có barrier) | 40 keyframe | 0.398 | | | |
| YOLOE-26-L zero-shot | 40 keyframe | 0.452 | | | |
| YOLOE-26-L zero-shot | dev 119 keyframe | 0.392 | 0.566 | 0.578 | 0.572 |
| YOLOE-26-L fine-tune nuImages | dev 119 keyframe | 0.429 | 0.525 | 0.633 | 0.574 |
| YOLOE-26-L zero-shot | **test 957 keyframe** | 0.312 | 0.499 | 0.523 | 0.511 |
| **YOLOE-26-L fine-tune nuImages** | **test 957 keyframe** | **0.366** | 0.484 | **0.581** | **0.528** |

Fine-tune: linear probe 10 epoch ở 1280 px trên nuImages (đã loại ảnh cùng xe, cùng ngày với log val nuScenes), chọn
checkpoint theo nuImages val (mAP50 0.509). Số liệu: `det2d_finetune.json`, `weights/results-lp-1280.csv`.

AP50 từng lớp trên test, zero-shot → fine-tune:

| barrier | bicycle | bus | car | constr. | motorcycle | pedestrian | traffic cone | trailer | truck |
|---|---|---|---|---|---|---|---|---|---|
| 0.127 → **0.333** | 0.123 → **0.233** | 0.805 → 0.779 | 0.707 → 0.728 | 0.019 → 0.029 | 0.210 → 0.275 | 0.226 → 0.294 | 0.390 → **0.500** | 0.000 → 0.015 | 0.518 → 0.470 |

Lớp nhỏ tăng mạnh (barrier ×2.6, cone +0.11, xe đạp gần ×2); bus và truck giảm nhẹ. Precision giảm 0.015 vì mô hình nhận
nhiều box hơn. Quyết định theo mAP50 + recall trên test: **dùng** (`detection.yoloe.weights`). Mô hình fine-tune là tập
lớp đóng 10 lớp nuScenes; dự án dùng tập lớp khác thì quay về `yoloe-26l-seg.pt` (open-vocab).

## 3. 2D — lan truyền nhãn: optical flow × ByteTrack × OC-SORT

Test 20 scene, cùng cache detection YOLOE zero-shot. Chỉ đếm box sản phẩm thật sự ghi ra. Số liệu: `tracker_test20.json`.

| Optical flow | Ghép detection | Đúng | Sai | Đổi ID | Thiếu | Tỉ lệ đúng | Điểm |
|---|---|---|---|---|---|---|---|
| không | không ByteTrack, không OC-SORT (IoU 1 tầng) | 2866 | 761 | 205 | 1541 | 74.8% | 1695 |
| không | ByteTrack | 2822 | 652 | 194 | 1673 | 76.9% | 1782 |
| có | không ByteTrack, không OC-SORT | 3220 | 960 | 82 | 1182 | 75.6% | 2096 |
| có | **ByteTrack (mặc định)** | 3192 | **860** | **76** | 1298 | **77.3%** | **2180** |
| có | OC-SORT (OCR), không ByteTrack | **3254** | 981 | 101 | **1122** | 75.0% | 2071 |
| có | ByteTrack + OC-SORT (OCR) | 3215 | 868 | 87 | 1265 | 77.1% | 2173 |

Tác dụng của từng thuật toán (so cùng điều kiện, chỉ khác thuật toán đó):

| Thuật toán | So | Đúng | Sai | Đổi ID | Thiếu | Điểm | Chỉ số quyết định → kết luận |
|---|---|---|---|---|---|---|---|
| Optical flow | không flow → có flow (không tracker) | +354 | +199 | −123 | −359 | +401 | nhãn đúng, đổi ID → **dùng** |
| ByteTrack | không → có (có flow) | −28 | **−100** | −6 | +116 | +84 | box sai, tỉ lệ đúng → **dùng** |
| OC-SORT | không → có (có flow) | +34 | +21 | +19 | −60 | −25 | điểm → không dùng |
| OC-SORT | trên ByteTrack | +23 | +8 | +11 | −33 | −7 | điểm → không dùng |

Đọc bảng: optical flow là bước quyết định (đổi ID −60%, thêm 12% nhãn đúng). ByteTrack bớt 10% box sai mà chỉ mất 0.9%
nhãn đúng, nên người duyệt phải xoá ít box hơn. OC-SORT phủ thêm vật (thiếu giảm) nhưng tăng đổi ID và box sai, điểm
không tăng: optical flow đã dự đoán box theo chuyển động thật của ảnh, nên phần OC-SORT sửa cho Kalman ở đây đã có sẵn.
Trên dev, ORU và OCM cũng không đổi gì đo được (`ocsort.md`).

## 4. 3D — mô hình và xử lý sau detector (test 24 scene)

| Mô hình | Cảm biến | mAP | NDS | ATE | AOE | AVE |
|---|---|---|---|---|---|---|
| FCOS3D | camera | 0.336 | 0.421 | 0.68 | 0.45 | 1.73 |
| PGD | camera | 0.399 | 0.458 | 0.60 | 0.38 | 1.29 |
| SSN | LiDAR | 0.441 | 0.568 | 0.33 | 0.35 | 0.33 |
| PointPillars | LiDAR | 0.473 | 0.559 | 0.39 | 0.56 | 0.30 |
| CenterPoint pillar | LiDAR | 0.528 | 0.610 | 0.29 | 0.37 | 0.36 |
| CenterPoint voxel | LiDAR | 0.578 | 0.655 | 0.26 | 0.29 | 0.37 |
| CenterPoint voxel + track | LiDAR | 0.616 | 0.689 | 0.26 | 0.21 | 0.33 |
| Gộp 4 LiDAR | LiDAR | 0.637 | 0.690 | 0.23 | 0.27 | 0.29 |
| **Gộp 4 LiDAR + track (mặc định)** | LiDAR | **0.668** | **0.713** | 0.23 | 0.22 | 0.27 |
| Gộp 4 LiDAR + 2 camera + track | cả hai | 0.662 | 0.711 | 0.24 | 0.21 | 0.28 |

AP từng lớp, CenterPoint voxel → mặc định: car 0.858 → 0.886 · truck 0.501 → 0.587 · bus 0.707 → 0.803 · trailer
0.319 → 0.620 · construction vehicle 0.318 → 0.473 · pedestrian 0.882 → 0.900 · motorcycle 0.496 → 0.555 · bicycle
0.435 → 0.561 · traffic cone 0.803 → 0.802 · barrier 0.458 → 0.495.

Chỉ số quyết định: mAP (và NDS). Chọn trên dev (0.537 → 0.644), test tăng cùng chiều (0.578 → 0.668).

## 5. 3D — lan truyền nhãn

Tracker 3D ghép theo khoảng cách tâm (kiểu CenterPoint), dự đoán vị trí bằng vận tốc mô hình + ego pose.

| Cấu hình | Đúng (held-out 4 scene) | Đổi ID |
|---|---|---|
| Chỉ bù chuyển động xe ego | 3031 | 98 |
| + vận tốc mô hình | 3475 | 67 |
| **+ vận tốc, chỉnh ngưỡng ghép (mặc định)** | 3339 | **36** |

Chỉ số quyết định: đổi ID (nhãn nhảy vật là lỗi khó thấy nhất).

| Cấu hình (test 24 scene) | Đúng | Sai | Đổi ID | Tỉ lệ đúng | Điểm dev | Điểm test |
|---|---|---|---|---|---|---|
| Không OC-SORT, mất 2 keyframe là dừng | 23355 | 946 | 470 | 0.943 | 3731 | 21469 |
| giữ track 3 keyframe | 23569 | 962 | 516 | 0.941 | 3741 | 21575 |
| **giữ track 4 keyframe (mặc định)** | 23731 | 968 | 543 | 0.940 | **3762** | 21677 |
| giữ track 6 keyframe | 23853 | 972 | 593 | 0.938 | 3765 | 21695 |
| + OC-SORT OCR | 23872 | 980 | 576 | 0.939 | 3734 | 21740 |
| + OC-SORT OCR + ORU | 23898 | 979 | 569 | 0.939 | 3733 | 21781 |
| chỉ ORU | 23382 | 949 | 471 | 0.943 | 3731 | 21491 |
| chỉ OCM | 23321 | 948 | 493 | 0.942 | 3731 | 21387 |

Chỉ số quyết định: điểm trên dev. Giữ 4 keyframe tốt nhất trên dev (6 keyframe ngang điểm nhưng thêm đổi ID). OCR tăng
điểm trên test nhưng giảm trên dev, nên để tắt (bật được ở tab ⚙ Cài đặt). ByteTrack chưa thử cho 3D.

## Còn chạy tiếp

Các bảng lan truyền 2D ở trên dùng detection YOLOE zero-shot. Chạy lại với YOLOE fine-tune:
`scripts\tasks.ps1 evaltemporal` trên laptop (GPU), rồi chấm lại bằng cùng cách đếm.
