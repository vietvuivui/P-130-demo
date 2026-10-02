# Tracking 2D — đánh giá từng phương pháp và các tổ hợp

Tài liệu này gom mọi phép đo về **lan truyền nhãn 2D** (tracker mang box người đã duyệt sang các frame sau): từng thuật
toán đơn lẻ (vận tốc không đổi, optical flow, ByteTrack, OC-SORT, BoT-SORT, DAM4SAM), rồi các tổ hợp, và trong mỗi tổ hợp
thuật toán nào làm nhiệm vụ gì. Số liệu thô: `tracker_test20.json`, `ocsort.json`, `temporal/*.json`,
`trackall*/dam4sam.json`, `trackall_heldout/dam4sam.json`.

## 1. Tracker gồm những bước nào, và mỗi thuật toán thay bước nào

Mọi cấu hình đều đi qua cùng một khung (`src/services/propagation.py: Tracker`); các phương pháp chỉ khác nhau ở cách làm
từng bước:

```
box người đã duyệt ở keyframe
        │
        ▼  (mỗi ảnh 12 Hz giữa hai keyframe)
① DỰ ĐOÁN  vật ở ảnh này nằm đâu?          vận tốc không đổi │ optical flow │ DAM4SAM (mask SAM 2.1)
        │
② GHÉP     box dự đoán ứng với box YOLO nào? IoU 1 lượt │ ByteTrack (2 lượt) │ BoT-SORT (ByteTrack + ngoại hình
        │                                     + bù chuyển động camera) │ OC-SORT (thêm OCR / ORU / OCM) │ không ghép
        ▼
③ QUẢN LÝ  giữ / sửa / dừng track            dừng sau 6 ảnh không khớp, ra khỏi khung, box quá nhỏ (chung cho mọi cấu hình)
        │
        ▼
box ghi ra ở keyframe sau = box YOLO đã ghép (mang ID + lớp của vật đã duyệt); không khớp thì không ghi
```

| Bước | Thuật toán | Nhiệm vụ cụ thể trong hệ thống |
|---|---|---|
| ① | Vận tốc không đổi | Dời box theo vận tốc đo ở các ảnh trước (cách cũ, không cần ảnh) |
| ① | Optical flow (OpenCV DIS) | Tính dịch chuyển từng pixel giữa hai ảnh, dời box theo các pixel bên trong nó; tự bù chuyển động camera |
| ① | DAM4SAM | SAM 2.1 tách mask của vật ở ảnh mới theo bộ nhớ hình dạng (có bộ nhớ nhận biết vật gây nhiễu); box dự đoán = hộp bao mask |
| ② | IoU một lượt | Ghép box dự đoán với box YOLO ≥ 0.10 chồng lấn nhiều nhất |
| ② | ByteTrack | Lượt 1 chỉ box YOLO ≥ 0.30; lượt 2 box 0.10–0.30 cho track còn thiếu (IoU chặt hơn). Box mờ không "cướp" track của vật rõ |
| ② | BoT-SORT | ByteTrack + chi phí ghép = min(1 − IoU, khoảng cách ngoại hình histogram màu HSV) + bù chuyển động camera (ORB + RANSAC) khi không có flow |
| ② | OC-SORT | OCR: track đang mất nhận lại theo box quan sát cuối; ORU: vận tốc theo đường quan sát cuối → mới; OCM: thưởng detection cùng hướng đi |
| ② | Không ghép | Chỉ với DAM4SAM: ghi thẳng hộp bao mask, bỏ qua YOLO ("DAM4SAM thuần") |

Vì box ghi ra luôn là box YOLO, chất lượng detector chặn trần của mọi cấu hình: DetA của các tổ hợp có ghép đều nằm
trong 0.46–0.48 (mục 4).

## 2. Cách đo

Thí nghiệm "keyframe hoàn hảo": nhãn gốc nuScenes ở keyframe 0, 5, 10… đóng vai nhãn người đã duyệt; tracker lan truyền
tối đa 10 keyframe; so với nhãn gốc cùng vật ở các keyframe sau. Chỉ đếm box sản phẩm thật sự ghi ra.

Có **hai đợt đo** với detector khác nhau, không so số tuyệt đối chéo đợt:

| Đợt | Detector (cache) | Tập | Chỉ số | Mục |
|---|---|---|---|---|
| A (9–10/2026, trước) | YOLOE-26-L zero-shot | dev 3 scene; test 20 scene (795 keyframe, 154 lần lan truyền) | nhãn đúng / box sai / đổi ID / mất dấu; điểm = đúng − sai − 2×đổi ID | 3.2–3.4 |
| B (02/10/2026) | YOLOE-26-L fine-tune nuImages | dev 3 scene; held-out 20 scene | như trên + TrackEval: HOTA, DetA, AssA, MOTA, IDF1, IDSW | 3.5–3.6, 4 |

Chỉ chọn cấu hình trên dev; test / held-out để báo cáo. Máy: RTX 4050 (đợt B), laptop GPU / VM (đợt A).

## 3. Từng phương pháp đơn lẻ

### 3.1 Dự đoán: vận tốc không đổi → optical flow (đợt A, test 20 scene, 154 lần, 1060 vật)

| `propagation.flow` | Nhãn đúng / box ra | Đổi ID | Mất dấu | Box sai | IoU TB | Giây |
|---|---|---|---|---|---|---|
| off: vận tốc không đổi | 2910 / 4531 (64.2%) | 265 | 1150 | 1356 | 0.508 | 51 |
| missing: flow chỉ ở ảnh chưa có detection | 3068 / 4484 (68.4%) | 181 | 1055 | 1235 | 0.541 | 51 |
| **always: flow ở mọi ảnh (mặc định)** | **3329 / 4693 (70.9%)** | **97** | **902** | 1267 | **0.567** | 214 |

Optical flow là bước quyết định: +14% nhãn đúng, đổi ID −63%, mất dấu −22%. Đổi lại là phần tốn thời gian nhất của lan
truyền (~1 s mỗi lần trên GPU).

### 3.2 Ghép: IoU một lượt → ByteTrack (đợt A)

| Dự đoán | Ghép | Test 20 scene: đúng | sai | đổi ID | thiếu | tỉ lệ đúng | điểm |
|---|---|---|---|---|---|---|---|
| vận tốc | IoU 1 lượt | 2866 | 761 | 205 | 1541 | 74.8% | 1695 |
| vận tốc | ByteTrack | 2822 | 652 | 194 | 1673 | 76.9% | 1782 |
| flow | IoU 1 lượt | 3220 | 960 | 82 | 1182 | 75.6% | 2096 |
| **flow** | **ByteTrack (mặc định)** | 3192 | **860** | **76** | 1298 | **77.3%** | **2180** |

ByteTrack bớt 10% box sai (người phải xoá) với giá 0.9% nhãn đúng, thời gian không đổi (chỉ là phép ghép trên box có sẵn).

### 3.3 Ghép: OC-SORT (đợt A, nền flow + ByteTrack)

| Cấu hình | Dev: đúng / sai / đổi ID / thiếu | Điểm dev | Test 20 scene | Điểm test |
|---|---|---|---|---|
| flow + ByteTrack (nền) | 750 / 156 / 20 / 336 | 554 | 3192 / 860 / 76 / 1298 | **2180** |
| + OCR | 759 / 162 / 19 / 323 | **559** | 3215 / 868 / 87 / 1265 | 2173 |
| + OCR + ORU | 758 / 161 / 21 / 323 | 555 | — | — |
| + OCR + ORU + OCM | 754 / 159 / 26 / 324 | 543 | — | — |
| OCR, không ByteTrack | — | — | 3254 / 981 / 101 / 1122 | 2071 |

Không dùng. OCR phủ thêm vật (thiếu −33) nhưng thêm đổi ID (+11) và box sai (+8). ORU / OCM không đổi gì đo được: optical
flow đã dự đoán theo chuyển động thật của ảnh, tức là đã có sẵn phần OC-SORT sửa cho Kalman. Trong 3D (dự đoán bằng vận
tốc của mô hình LiDAR) kết luận cũng vậy; thứ có ích ở 3D là giữ track 4 keyframe thay vì 2 (`ocsort.md`).

### 3.4 Ghép: BoT-SORT (đợt B)

| Tập | Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | IDF1 | Giây | Ngoại hình đổi quyết định |
|---|---|---|---|---|---|---|---|---|---|
| dev 3 scene | flow + ByteTrack | 846 | 209 | 32 | 201 | 0.580 | 0.757 | 62–81 | — |
| dev 3 scene | flow + BoT-SORT | 846 | 209 | 28 | 203 | 0.580 | 0.758 | 98 | 8 / 1770 lần ghép |
| held-out 20 | flow + ByteTrack | 3438 | 1161 | 171 | 853 | 0.563 | 0.705 | 731 | — |
| held-out 20 | flow + BoT-SORT | 3460 | 1146 | 161 | 853 | 0.564 | 0.709 | 1090 | 83 / 9981 lần ghép |

Gần như không đổi gì khi đã có optical flow: phần bù chuyển động camera của BoT-SORT chỉ chạy khi không có flow, còn
phần ngoại hình (histogram màu, không phải mạng ReID) chỉ đổi dưới 1% quyết định ghép. Thời gian +50%.

### 3.5 Dự đoán: DAM4SAM (đợt B, dev 3 scene, SAM 2.1 hiera-large, stride 3)

| Dự đoán | Ghép | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| flow | ByteTrack (nền) | 846 | 209 | 32 | 201 | 0.580 | 0.481 | 0.733 | 0.757 | 62 |
| DAM4SAM | không ghép (thuần) ¹ | 1007 | 691 | 83 | 0 | 0.532 | 0.428 | 0.700 | 0.685 | 1029 |
| DAM4SAM | ByteTrack | 849 | 261 | 20 | 200 | 0.574 | 0.470 | 0.728 | 0.740 | 848 |
| DAM4SAM | BoT-SORT | 856 | 248 | 18 | 196 | 0.578 | 0.475 | 0.733 | 0.747 | 731 |
| flow, DAM4SAM chỉ cho vật mất detection | ByteTrack | 859 | 255 | 67 | 173 | 0.562 | 0.476 | 0.695 | 0.730 | 506 |

¹ Đo trước khi sửa ngưỡng dừng theo stride (bên dưới); cấu hình này không dùng ngưỡng theo detection nên vẫn so được.

- **DAM4SAM thuần** (bỏ YOLO) tệ nhất: không có detection thì không biết dừng khi vật khuất (0 mất dấu nhưng 691 box sai),
  hai tracker có thể bám cùng một vật, và không có lớp / score cho QA Agent. Bước ghép với YOLO là bắt buộc.
- **DAM4SAM + ghép** ngang flow + ByteTrack trên dev, ít đổi ID hơn (18–20 so với 32) nhưng nhiều box sai hơn.
- **Cấu hình lai** (SAM chỉ gọi khi vật mất detection) đổi ID gấp đôi: SAM được gọi "nguội" sau một quãng không theo dõi,
  box nhảy sang hộp bao mask rồi khớp nhầm detection của vật khác (giả thuyết, chưa kiểm chứng).
- **Stride 3** (keyframe + 1 ảnh giữa hai keyframe) giảm thời gian 3–4 lần so với mọi ảnh 12 Hz (bản chưa tăng tốc:
  20–24 phút một cấu hình trên scene-0035, nay 3 phút). Dùng chung image encoder cho mọi vật trong ảnh + autocast:
  33.5 → 9.5 giây trên phép thử một vật.
- **Ngưỡng dừng theo stride**: track dừng sau 6 ảnh không khớp; với stride 3, 6 ảnh là 1,5 giây thay vì 0,5 giây nên
  track sống quá lâu. Sửa (chia ngưỡng theo stride): DAM4SAM + ByteTrack box sai 418 → 261, HOTA 0.562 → 0.574;
  DAM4SAM + BoT-SORT 371 → 248, 0.571 → 0.578.

## 4. Các tổ hợp (ensemble) — bảng tổng

Cột "Dự đoán" và "Ghép" ghi rõ thuật toán nào làm bước nào; bước ③ (dừng track) giống nhau ở mọi dòng.

### Held-out 20 scene, 795 keyframe (đợt B)

| Luồng trên UI | Dự đoán (①) | Ghép (②) | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Nhanh (mặc định)** | optical flow | ByteTrack | 3438 | 1161 | 171 | 853 | 0.563 | 0.457 | 0.750 | 0.453 | 0.705 | 731 |
| — | optical flow | BoT-SORT | 3460 | 1146 | 161 | 853 | 0.564 | 0.459 | 0.751 | 0.459 | 0.709 | 1090 |
| **Chính xác** | DAM4SAM | BoT-SORT | **3678** | 1315 | **155** | **639** | **0.573** | **0.472** | **0.761** | **0.467** | **0.725** | 2456 |

### Dev 3 scene (đợt B)

| Dự đoán (①) | Ghép (②) | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|
| optical flow | ByteTrack | 846 | 209 | 32 | 201 | **0.580** | **0.757** | 62 |
| optical flow | BoT-SORT | 846 | 209 | 28 | 203 | 0.580 | 0.758 | 98 |
| DAM4SAM | BoT-SORT | 856 | 248 | 18 | 196 | 0.578 | 0.747 | 731 |
| DAM4SAM | ByteTrack | 849 | 261 | 20 | 200 | 0.574 | 0.740 | 848 |
| flow + DAM4SAM (lai) | ByteTrack | 859 | 255 | 67 | 173 | 0.562 | 0.730 | 506 |
| DAM4SAM | không ghép | 1007 | 691 | 83 | 0 | 0.532 | 0.685 | 1029 |

### Test 20 scene (đợt A, detector zero-shot)

| Dự đoán (①) | Ghép (②) | Nhãn đúng | Box sai | Đổi ID | Mất dấu | Tỉ lệ đúng | Điểm |
|---|---|---|---|---|---|---|---|
| vận tốc | IoU 1 lượt | 2866 | 761 | 205 | 1541 | 74.8% | 1695 |
| vận tốc | ByteTrack | 2822 | 652 | 194 | 1673 | 76.9% | 1782 |
| optical flow | IoU 1 lượt | 3220 | 960 | 82 | 1182 | 75.6% | 2096 |
| optical flow | ByteTrack | 3192 | 860 | 76 | 1298 | 77.3% | **2180** |
| optical flow | OC-SORT (OCR) | 3254 | 981 | 101 | 1122 | 75.0% | 2071 |
| optical flow | ByteTrack + OCR | 3215 | 868 | 87 | 1265 | 77.1% | 2173 |

## 5. Kết luận

- **Mặc định: optical flow + ByteTrack** ("Nhanh"). Chạy CPU, không cần cài gì thêm; flow là bước quyết định, ByteTrack
  bớt 10% box sai gần như miễn phí.
- **DAM4SAM + BoT-SORT** ("Chính xác", người dùng tự chọn khi có GPU): trên held-out hơn mặc định HOTA +0.010, IDF1
  +0.020, nhãn đúng +7%, mất dấu −25%; đổi lại box sai +13% và chậm hơn 3,4 lần. Tỉ lệ box đúng trên box ghi ra gần
  như không đổi (72,1% → 71,4%): DAM4SAM **giữ được nhiều vật hơn** qua đoạn khó chứ không chính xác hơn trên từng box.
  Trên dev hai luồng ngang nhau (0.580 / 0.578), nên mức hơn này nhỏ và chưa có khoảng tin cậy theo scene.
- **OC-SORT, BoT-SORT thêm vào optical flow**: không đổi gì đáng kể, vì phần hai thuật toán này sửa cho Kalman filter
  (bù chuyển động camera, dựa vào quan sát) thì optical flow đã làm ở mức từng pixel.
- **Vì sao các phương pháp mới không hơn nhiều**: (1) baseline không phải ByteTrack gốc mà là ByteTrack + optical flow;
  (2) bài toán dễ hơn benchmark MOT ở đúng chỗ các phương pháp mới giỏi: track khởi tạo từ box người duyệt, chỉ 5 giây,
  ảnh 12 Hz; (3) box ghi ra là box YOLO nên detector chặn trần (DetA 0.46–0.48 ở mọi cấu hình); (4) bản cài rút gọn:
  BoT-SORT dùng histogram màu thay mạng ReID, DAM4SAM là tracker một vật chạy ở stride 3; (5) tham số baseline đã chỉnh
  trên chính dữ liệu này, các phương pháp mới chạy tham số mặc định.

## 6. Chưa đo

- Kết quả theo từng scene / khoảng tin cậy trên held-out.
- DAM4SAM + ByteTrack, DAM4SAM thuần, cấu hình lai trên held-out; DAM4SAM ở stride 1 trên cả 3 scene; model nhỏ (`sam21pp-T`).
- BoT-SORT với mạng ReID thật; chỉnh tham số cho các cấu hình DAM4SAM.
- Detector sau khi fine-tune thêm trên RTX 3090 (nâng trần DetA là cách chắc nhất để nâng HOTA).

## Chạy lại

```
# held-out 20 scene, 3 cấu hình (khoảng 70 phút trên RTX 4050 + detect ảnh 12 Hz lần đầu)
python tools2d\dam4sam.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_heldout --out eval\results\trackall_heldout --stride 3 --configs flow+byte flow+botsort dam4sam+botsort
# dev 3 scene, mọi cấu hình
python tools2d\dam4sam.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_dev --out eval\results\trackall_v2 --stride 3 --scenes scene-0035 scene-0097 scene-0101
# đợt A (flow / ByteTrack / OC-SORT): scripts\tasks.ps1 evaltemporal ; tools2d\ocsort.py (xem ocsort.md)
```
