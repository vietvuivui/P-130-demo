# Bảng so sánh các phương pháp đã dùng (2D và 3D)

Cập nhật 2026-10-01. Mọi số chất lượng đều so với **nhãn gốc nuScenes**. Cách làm được chọn trên **dev** (3 scene demo) và
báo cáo trên **test** (các scene val khác, chưa dùng để chọn gì):

- 3D: 24 scene, 957 keyframe.
- Detector 2D: 24 scene, 957 keyframe CAM_FRONT.
- Lan truyền 2D: 20 scene, 795 keyframe, 154 lần lan truyền.

Chỉ số dùng trong bảng:

- **mAP50** (2D): AP trung bình của 10 lớp; box tính là đúng khi IoU ≥ 0.5. **P / R**: precision / recall ở ngưỡng của
  sản phẩm.
- **mAP / NDS** (3D): chuẩn nuScenes. mAP ghép box theo khoảng cách tâm; NDS gộp thêm sai số vị trí, kích thước, hướng
  và vận tốc.
- **Lan truyền nhãn**: nhãn gốc ở keyframe 0, 5, 10… đóng vai nhãn người đã duyệt, rồi lan truyền tiếp tối đa 10
  keyframe.
  - *Đúng*: box ghi ra nằm đúng vật (IoU ≥ 0.5, cùng vật).
  - *Sai*: box không trùng vật nào.
  - *Đổi ID*: box nhảy sang vật khác.
  - *Thiếu*: vật còn trong ảnh nhưng không có box.
  - **Điểm = đúng − sai − 2 × đổi ID.** Đổi ID tính gấp đôi vì box mang lớp và kích thước của vật cũ, nên người duyệt
    khó phát hiện hơn box sai hẳn.
- **Thời gian inference**: đo trên ba loại máy.
  - **GPU** = laptop RTX 4050.
  - **CPU** = 2 luồng x86.
  - **VM** = máy ảo CPU trên laptop, chỉ dùng để so tương đối giữa các cấu hình trong cùng một lần chạy.

  Số ghi kèm loại máy. Chi tiết ở mục 6.

## 1. Tổng hợp: mỗi phương pháp một dòng

| # | Phần | Phương pháp | Chỉ số quyết định | Không dùng | Có dùng | Thay đổi | Thời gian thêm / bớt | Dùng? |
|---|---|---|---|---|---|---|---|---|
| 1 | 2D | YOLOE-26-L thay YOLO-World | mAP50 (40 keyframe) | 0.311 | 0.452 | +0.141 | 3.75 → 3.31 s/ảnh, CPU (−12%) | ✅ |
| 2 | 2D | Fine-tune YOLOE-26-L trên nuImages, linear probe | mAP50 test | 0.312 | 0.366 | +0.054 (R +0.058, P −0.015) | không đổi: cùng kiến trúc, cùng 1280 px; GPU 0.076 s/ảnh | thay bằng 2b |
| 2b | 2D | **Fine-tune toàn mạng YOLOE-26-L trên nuImages** (mặc định từ 04/10) | mAP50 test | 0.366 (linear probe) | 0.589 | +0.223 (R +0.226, P +0.003); so với zero-shot +0.277 | không đổi: cùng kiến trúc, cùng 1280 px; GPU 0.069 s/ảnh | ✅ mới |
| 2c | 2D | Gộp box 3D vào nhãn 2D, box chỉ camera không có điểm LiDAR ×0.3 (đo 6 camera, 04/10) | mAP50 / box phải sửa (test 5742 ảnh) | 0.560 / 23432 (chỉ detector ảnh) | 0.672 / 15184 | +0.112 / −35% (riêng hệ số ×0.3: phải sửa 17348 → 15184) | cần chạy 4 mô hình LiDAR; phần gộp chạy CPU | ✅ mới |
| 3 | 2D | Ngưỡng giữ box 0.30 | precision (40 keyframe) | 0.34 | 0.51 | +0.17 | 0 | ✅ |
| 4 | 2D | TTA: đổi prompt, lật ảnh, ảnh 1600 px | mAP50 dev | 0.395 | 0.365–0.392 | giảm | thêm một lần suy luận cho mỗi biến thể (chưa đo) | ❌ |
| 5 | 2D | Optical flow khi lan truyền | nhãn đúng / đổi ID (test) | 2866 / 205 | 3220 / 82 | +12% / −60% | 0.33 → 1.39 s/lần lan truyền, GPU (×4.2) | ✅ |
| 6 | 2D | ByteTrack (ghép 2 tầng) | box sai (test) | 960 | 860 | −10% (đúng −0.9%) | 67.6 → 67.4 s trên dev, cùng máy (≈ 0) | ✅ |
| 7 | 2D | OC-SORT (OCR), không ByteTrack | điểm (test) | 2096 | 2071 | −25 | 16.2 → 16.5 s, VM (< 2%) | ❌ |
| 8 | 2D | OC-SORT (OCR) thêm vào ByteTrack | điểm (test) | 2180 | 2173 | −7 | 16.4 → 16.2 s, VM (< 2%) | ❌ |
| 9 | 2D | OC-SORT ORU / OCM | điểm dev | 554 | 550 / 549 | ≈ 0 | ≈ 0: vài phép tính trên box, không thêm suy luận | ❌ |
| 10 | 2D | Optical flow cho QA temporal | lỗi lọt qua duyệt theo lô | 1632 | 2082 | +28% | 0.044 → 0.143 s/keyframe, GPU (+0.10 s) | ❌ |
| 11 | 2D | Tính lại score theo sweep (Seq-NMS) | box phải xem tay | 1707 | 648 | −62% (vật sót +9%) | 0.044 → 0.047 s/keyframe, GPU | tuỳ chọn |
| 12 | 2D | Nhận box sweep score thấp (ý ByteTrack) | lỗi lọt qua duyệt theo lô | — | tăng | xấu hơn | ≈ 0: dùng lại detection đã có | ❌ |
| 13 | 3D | CenterPoint voxel (mô hình đơn tốt nhất) | mAP test | 0.473 (PointPillars) | 0.578 | +0.105 | 0.85 → 0.62 s/keyframe, GPU | ✅ |
| 14 | 3D | Gộp 4 mô hình LiDAR (WBF) | mAP test | 0.578 | 0.637 | +0.059 | 0.62 → 2.55 s/keyframe: 4 mô hình 2.47 s GPU + gộp 0.08 s CPU | ✅ |
| 15 | 3D | Tinh chỉnh theo track | mAP / NDS test | 0.637 / 0.690 | 0.668 / 0.713 | +0.031 / +0.023 | +0.003–0.006 s/keyframe, CPU | ✅ |
| 16 | 3D | Thêm 2 mô hình camera vào ensemble | mAP test | 0.668 | 0.662 | −0.006 | 2.47 → 7.90 s/keyframe, GPU (×3.2) | ❌ |
| 17 | 3D | Kiểm tra chéo 3D bằng 2D (chấm lại / xoá box) | box sửa đúng / làm hỏng (dev) | — | 14 / 39 | hỏng nhiều hơn | dùng lại detection 2D đã có cho kiểm chứng | ❌ |
| 18 | 3D | VESPA: box 3D từ mask 2D + LiDAR | mAP dev | 0.644 | 0.627 | −0.017 | chưa đo | ❌ |
| 19 | 3D | VESPA: hướng theo chuyển động, cỡ theo lớp | NDS test | 0.713 | 0.712–0.713 | ≈ 0 | ≈ 0 | ❌ |
| 20 | 3D | Lan truyền theo vận tốc mô hình | đổi ID (held-out 4 scene) | 98 | 36 | −63% | ≈ 0: dùng vận tốc mô hình đã có | ✅ |
| 21 | 3D | Giữ track 4 keyframe thay vì 2 (ý OC-SORT) | điểm dev; nhãn đúng test | 3731; 23355 | 3762; 23731 | +31; +1.6% | ≈ 0 | ✅ |
| 22 | 3D | OC-SORT OCR | điểm dev | 3762 | 3734 | −28 (test +63) | ≈ 0 | ❌ (bật được) |
| 23 | 3D | OC-SORT ORU / OCM | điểm test | 21469 | 21491 / 21387 | ≈ 0 | ≈ 0 | ❌ |

## 2. 2D — detector

| Detector | Tập | mAP50 | P | R | F1 | Thời gian / ảnh (1280 px) |
|---|---|---|---|---|---|---|
| YOLO-World (từ vựng COCO, không có barrier) | 40 keyframe | 0.311 | | | | 3.75 s CPU |
| YOLO26 (tập lớp COCO, không có barrier) | 40 keyframe | 0.398 | | | | 2.14 s CPU |
| YOLOE-26-L zero-shot | 40 keyframe | 0.452 | | | | 3.31 s CPU |
| YOLOE-26-L zero-shot | dev 119 keyframe | 0.392 | 0.566 | 0.578 | 0.572 | |
| YOLOE-26-L linear probe nuImages | dev 119 keyframe | 0.429 | 0.525 | 0.633 | 0.574 | |
| YOLOE-26-L fine-tune toàn mạng nuImages | dev 119 keyframe | 0.616 | 0.472 | 0.825 | 0.600 | |
| YOLOE-26-L zero-shot | **test 957 keyframe** | 0.312 | 0.499 | 0.523 | 0.511 | như dòng dưới (cùng kiến trúc) |
| YOLOE-26-L linear probe nuImages | **test 957 keyframe** | 0.366 | 0.484 | 0.581 | 0.528 | 0.076 s GPU (957 ảnh / 72.7 s, batch 8) |
| **YOLOE-26-L fine-tune toàn mạng nuImages (mặc định)** | **test 957 keyframe** | **0.589** | 0.487 | **0.806** | **0.607** | **0.069 s GPU** (957 ảnh / 66.5 s, batch 8) |

Bản fine-tune toàn mạng: 30 epoch ở 1280 px trên nuImages (RTX 3090), mở cả backbone, neck và nhánh box; checkpoint chọn
theo nuImages val (mAP50 0.782). Chấm trên laptop ngày 04/10, cùng 957 keyframe test (`weights/results-full-1280.csv`).

Bản linear probe: 10 epoch ở 1280 px trên nuImages, đã loại ảnh cùng xe, cùng ngày với log val nuScenes.
Checkpoint được chọn theo nuImages val (mAP50 0.509). Linear probe chỉ học lại đầu phân lớp nên kiến trúc và lượng tính
toán không đổi. Lần chấm zero-shot trên GPU đọc lại detection từ cache, vì vậy không có số thời gian riêng. Số liệu:
`det2d_finetune.json`, `compare/speed.json`, `weights/results-lp-1280.csv`.

AP50 từng lớp trên test, zero-shot → linear probe → fine-tune toàn mạng:

| barrier | bicycle | bus | car | constr. | motorcycle | pedestrian | traffic cone | trailer | truck |
|---|---|---|---|---|---|---|---|---|---|
| 0.127 → 0.333 → **0.676** | 0.123 → 0.233 → **0.630** | 0.805 → 0.779 → **0.887** | 0.707 → 0.728 → **0.853** | 0.019 → 0.029 → **0.111** | 0.210 → 0.275 → **0.541** | 0.226 → 0.294 → **0.563** | 0.390 → 0.500 → **0.798** | 0.000 → 0.015 → **0.224** | 0.518 → 0.470 → **0.611** |

Kết quả theo lớp:

- **Linear probe:** lớp nhỏ tăng mạnh (barrier ×2.6, traffic cone +0.11, xe đạp gần ×2); bus và truck giảm nhẹ;
  precision giảm 0.015 vì mô hình nhận nhiều box hơn.
- **Fine-tune toàn mạng:** tăng ở cả 10 lớp so với cả hai bản trước. Recall 0.523 → 0.806; recall theo chiều cao vật:
  dưới 32 px 26% → 69%, 32–64 px 47% → 83%, 64–128 px 56% → 87%, từ 128 px 63% → 79%.
- **Precision gần như không đổi** (0.499 → 0.487 so với zero-shot): ở ngưỡng 0.30 người duyệt vẫn phải xoá khoảng một
  nửa số box.
- **Còn yếu:** construction vehicle 0.111 và trailer 0.224 (trailer chỉ có 22 vật trên test).

Quyết định dựa trên mAP50 và recall trên test: **dùng bản fine-tune toàn mạng** (`detection.yoloe.weights`). Hai bản
fine-tune chỉ biết tập lớp đóng gồm 10 lớp nuScenes. Dự án dùng tập lớp khác thì vào tab ⚙ Cài đặt > "Mô hình phát hiện
2D" chọn Original (`yoloe-26l-seg.pt`, open-vocab); ba lựa chọn trên UI là Fine-tune full / Fine-tune mini (linear
probe) / Original.

## 3. 2D — lan truyền nhãn: optical flow × ByteTrack × OC-SORT

Test 20 scene, dùng chung cache detection YOLOE zero-shot. Chỉ đếm những box sản phẩm thật sự ghi ra. Số liệu:
`tracker_test20.json`.

| Optical flow | Ghép detection | Đúng | Sai | Đổi ID | Thiếu | Tỉ lệ đúng | Điểm |
|---|---|---|---|---|---|---|---|
| không | không ByteTrack, không OC-SORT (IoU 1 tầng) | 2866 | 761 | 205 | 1541 | 74.8% | 1695 |
| không | ByteTrack | 2822 | 652 | 194 | 1673 | 76.9% | 1782 |
| có | không ByteTrack, không OC-SORT | 3220 | 960 | 82 | 1182 | 75.6% | 2096 |
| có | **ByteTrack (mặc định)** | 3192 | **860** | **76** | 1298 | **77.3%** | **2180** |
| có | OC-SORT (OCR), không ByteTrack | **3254** | 981 | 101 | **1122** | 75.0% | 2071 |
| có | ByteTrack + OC-SORT (OCR) | 3215 | 868 | 87 | 1265 | 77.1% | 2173 |

Tác dụng của từng thuật toán (so trong cùng điều kiện, chỉ khác thuật toán đó):

| Thuật toán | So | Đúng | Sai | Đổi ID | Thiếu | Điểm | Thời gian | Chỉ số quyết định → kết luận |
|---|---|---|---|---|---|---|---|---|
| Optical flow | không flow → có flow (không tracker) | +354 | +199 | −123 | −359 | +401 | ×4.2 | nhãn đúng, đổi ID → **dùng** |
| ByteTrack | không → có (có flow) | −28 | **−100** | −6 | +116 | +84 | ≈ 0 | box sai, tỉ lệ đúng → **dùng** |
| OC-SORT | không → có (có flow) | +34 | +21 | +19 | −60 | −25 | < 2% | điểm → không dùng |
| OC-SORT | thêm vào ByteTrack | +23 | +8 | +11 | −33 | −7 | < 2% | điểm → không dùng |

Đọc bảng:

- **Optical flow là bước quyết định:** đổi ID giảm 60%, nhãn đúng tăng 12%. Đổi lại, đây cũng là bước tốn thời gian
  nhất của phần lan truyền.
- **ByteTrack bớt 10% box sai** mà chỉ mất 0.9% nhãn đúng, gần như không tốn thêm thời gian. Người duyệt phải xoá ít box
  hơn.
- **OC-SORT phủ thêm vật** (số vật thiếu giảm), nhưng làm tăng đổi ID và box sai nên điểm không tăng. Lý do: optical flow
  đã dự đoán box theo chuyển động thật của ảnh, nên phần OC-SORT sửa cho Kalman ở đây đã có sẵn.
- **ORU và OCM:** trên dev cũng không đổi được gì đo được (`ocsort.md`).

## 4. 3D — mô hình và xử lý sau detector (test 24 scene)

| Mô hình | Cảm biến | mAP | NDS | ATE | AOE | AVE | s / keyframe (GPU) | VRAM đỉnh |
|---|---|---|---|---|---|---|---|---|
| FCOS3D | camera | 0.336 | 0.421 | 0.68 | 0.45 | 1.73 | 2.72 | 0.55 GB |
| PGD | camera | 0.399 | 0.458 | 0.60 | 0.38 | 1.29 | 2.71 | 0.56 GB |
| SSN | LiDAR | 0.441 | 0.568 | 0.33 | 0.35 | 0.33 | 0.59 | 0.57 GB |
| PointPillars | LiDAR | 0.473 | 0.559 | 0.39 | 0.56 | 0.30 | 0.85 | 1.74 GB |
| CenterPoint pillar | LiDAR | 0.528 | 0.610 | 0.29 | 0.37 | 0.36 | **0.41** | 0.48 GB |
| CenterPoint voxel | LiDAR | 0.578 | 0.655 | 0.26 | 0.29 | 0.37 | 0.62 | 0.50 GB |
| CenterPoint voxel + track | LiDAR | 0.616 | 0.689 | 0.26 | 0.21 | 0.33 | 0.63 (track +0.005) | |
| Gộp 4 LiDAR | LiDAR | 0.637 | 0.690 | 0.23 | 0.27 | 0.29 | 2.55 (gộp +0.08) | |
| **Gộp 4 LiDAR + track (mặc định)** | LiDAR | **0.668** | **0.713** | 0.23 | 0.22 | 0.27 | **2.56** | |
| Gộp 4 LiDAR + 2 camera + track | cả hai | 0.662 | 0.711 | 0.24 | 0.21 | 0.28 | ~8.0 | |

Ghi chú về thời gian:

- Thời gian đo trên 1076 keyframe, đã gồm cả nạp dữ liệu (`eval/results/det3d/*/run.json`).
- Ensemble = tổng thời gian 4 mô hình chạy lần lượt.
- Gộp và tinh chỉnh theo track là numpy chạy trên CPU, không có suy luận mô hình. Đo trên máy ảo 2 nhân (i7-12650H),
  239 keyframe, lần chạy thứ hai: gộp 4 mô hình 73–85 ms/keyframe, track 2–6 ms/keyframe.

AP từng lớp, CenterPoint voxel → mặc định:

| Lớp | CenterPoint voxel | Mặc định |
|---|---|---|
| car | 0.858 | 0.886 |
| truck | 0.501 | 0.587 |
| bus | 0.707 | 0.803 |
| trailer | 0.319 | 0.620 |
| construction vehicle | 0.318 | 0.473 |
| pedestrian | 0.882 | 0.900 |
| motorcycle | 0.496 | 0.555 |
| bicycle | 0.435 | 0.561 |
| traffic cone | 0.803 | 0.802 |
| barrier | 0.458 | 0.495 |

Chỉ số quyết định là mAP (kèm NDS). Cấu hình được chọn trên dev (0.537 → 0.644), và test tăng cùng chiều (0.578 → 0.668).
Ensemble chậm gấp khoảng 4 lần mô hình đơn (2.56 so với 0.62 s/keyframe) nhưng vẫn chấp nhận được với việc gán nhãn
offline. Thêm camera thì chậm gấp 3.2 lần nữa mà còn kém đi.

## 5. 3D — lan truyền nhãn

Tracker 3D ghép box theo khoảng cách tâm (kiểu CenterPoint) và dự đoán vị trí bằng vận tốc của mô hình cộng ego pose.
Bước này chỉ dùng dự đoán đã có, không chạy lại mô hình.

| Cấu hình | Đúng (held-out 4 scene) | Đổi ID |
|---|---|---|
| Chỉ bù chuyển động xe ego | 3031 | 98 |
| + vận tốc mô hình | 3475 | 67 |
| **+ vận tốc, chỉnh ngưỡng ghép (mặc định)** | 3339 | **36** |

Chỉ số quyết định: đổi ID, vì nhãn nhảy sang vật khác là lỗi khó thấy nhất.

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

Chỉ số quyết định: điểm trên dev.

- **Giữ track 4 keyframe** cho điểm dev tốt nhất. Giữ 6 keyframe ngang điểm nhưng thêm đổi ID.
- **OCR** tăng điểm trên test nhưng giảm trên dev, nên để tắt. Có thể bật ở tab ⚙ Cài đặt.
- **ByteTrack** chưa thử cho 3D.

## 6. Thời gian inference

**Một keyframe qua toàn bộ pipeline.** Đo bằng `scripts\tasks.ps1 profile`, không dùng cache:

| Bước | GPU laptop | CPU |
|---|---|---|
| 2D: detect keyframe + 4 sweep, chiếu LiDAR, QA Agent | ~1 s / keyframe | detect 17.7 s / keyframe |
| 3D: 4 mô hình LiDAR, kiểm chứng bằng 6 camera | ~4 s / keyframe | không chạy được (cần op CUDA) |
| 3D: gộp 4 mô hình + tinh chỉnh theo track | — | ~0.08 s / keyframe (máy ảo 2 nhân) |

**Xử lý sau detector**, khi detection đã có trong cache. Test 20 scene, 795 keyframe, GPU
(`temporal/gpu_heldout20.json`):

| Cấu hình | s / keyframe |
|---|---|
| Mặc định | 0.044 |
| + optical flow cho QA temporal | 0.143 |
| + tính lại score theo sweep (mean) | 0.047 |

**Lan truyền nhãn 2D.** Test 20 scene, 154 lần lan truyền; mỗi lần đi tối đa 10 keyframe, khoảng 60 ảnh 12 Hz:

| Cấu hình | Tổng | Mỗi lần lan truyền | Máy |
|---|---|---|---|
| Không optical flow | 51.4 s | 0.33 s | GPU |
| Optical flow (mặc định) | 213.9 s | 1.39 s | GPU |
| Không tracker / ByteTrack, có flow (dev) | 67.6 / 67.4 s | — | cùng một máy (`temporal/bytetrack.json`) |
| ByteTrack / không tracker / OCR / OCR không ByteTrack (scene-0093, flow đã tính sẵn) | 16.4 / 16.2 / 16.2 / 16.5 s | — | VM |
| Tính optical flow cho scene-0093 | 43.5 s | — | VM |

Kết luận về thời gian:

- **Optical flow** là phần đắt nhất của lan truyền: khoảng 1 s mỗi lần lan truyền trên GPU.
- **ByteTrack và OC-SORT** chỉ là phép ghép trên box có sẵn, nên khác nhau dưới 2%. Thời gian không phải lý do chọn hay
  bỏ chúng. Quyết định dựa trên chất lượng (mục 3).
