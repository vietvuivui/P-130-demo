# OC-SORT trong tracker lan truyền nhãn (2D và 3D)

OC-SORT (Cao et al., CVPR 2023, github.com/noahcao/OC_SORT) sửa ba điểm yếu của tracker dự đoán bằng Kalman / vận tốc
không đổi khi vật bị che:

- **OCR** (observation-centric recovery): sau các lượt ghép, track đang mất ghép thêm với detection còn thừa theo **box
  quan sát cuối**, không theo box dự đoán đã trôi;
- **ORU** (re-update): ghép lại sau khi mất thì lấy vận tốc theo đường nối quan sát cuối → quan sát mới;
- **OCM** (momentum): cộng điểm cho detection nằm cùng hướng đi đã quan sát.

Code: `src/services/propagation.py` (2D), `src/services/propagation3d.py` (3D), cấu hình `propagation.oc_*`,
`propagation3d.oc_*`, bật / tắt được trên tab ⚙ Cài đặt. Số liệu thô: `ocsort.json`.

## Cách đo

Thí nghiệm "keyframe hoàn hảo" như các lần trước: nhãn gốc ở keyframe 0, 5, 10… đóng vai nhãn người đã duyệt, lan truyền
tối đa 10 keyframe, so với nhãn gốc cùng vật. Chọn cấu hình trên **dev** (3 scene demo), báo cáo trên **test**.

Điểm tổng để quyết định: **nhãn đúng − box sai − 2 × đổi ID**. Đổi ID tính gấp đôi vì box nằm trên vật khác nhưng mang
lớp / kích thước của vật cũ, người khó phát hiện hơn box sai hẳn.

2D chỉ đếm box **sản phẩm thật sự ghi ra** (track khớp detection ở keyframe đích, `emit_coasting: false`).

## 2D (YOLOE + optical flow + ghép kiểu ByteTrack)

| Cấu hình | Dev: đúng / sai / đổi ID / thiếu | Điểm dev | Test 20 scene: đúng / sai / đổi ID / thiếu | Điểm test |
|---|---|---|---|---|
| Hiện tại | 750 / 156 / 20 / 336 | 554 | 3192 / 860 / 76 / 1298 | **2180** |
| + OCR (nhận lại theo quan sát cuối) | 759 / 162 / 19 / 323 | **559** | 3215 / 868 / 87 / 1265 | 2173 |
| + OCR + ORU | 758 / 161 / 21 / 323 | 555 | — | — |
| + OCR + ORU + OCM | 754 / 159 / 26 / 324 | 543 | — | — |
| chỉ ORU | 749 / 155 / 22 / 336 | 550 | — | — |
| chỉ OCM | 749 / 154 / 23 / 335 | 549 | — | — |
| giữ track 18 ảnh (không OCR) | 762 / 162 / 21 / 321 | 558 | — | — |

Test 20 scene = 795 keyframe, 154 lần lan truyền, chạy trên cache detection của laptop.

**Không áp dụng.** OCR thêm 23 nhãn đúng (+0.7%) và bớt 33 vật thiếu nhãn trên test, nhưng thêm 11 lần đổi ID và 8 box
sai, nên điểm tổng không tăng. ORU và OCM không đổi gì đo được: optical flow đã dự đoán box theo chuyển động thật của ảnh
(chính là tinh thần "dựa vào quan sát" của OC-SORT), nên phần OC-SORT sửa cho Kalman ở đây đã có sẵn.

## 3D (ensemble 4 mô hình LiDAR + track)

| Cấu hình | Dev: đúng / sai / đổi ID | Điểm dev | Test 24 scene: đúng / sai / đổi ID | Tỉ lệ đúng test | Điểm test |
|---|---|---|---|---|---|
| Hiện tại (mất 2 keyframe là dừng) | 4074 / 131 / 106 | 3731 | 23355 / 946 / 470 | 0.943 | 21469 |
| giữ track 3 keyframe | 4093 / 132 / 110 | 3741 | 23569 / 962 / 516 | 0.941 | 21575 |
| **giữ track 4 keyframe (mới)** | 4126 / 132 / 116 | **3762** | **23731** / 968 / 543 | 0.940 | **21677** |
| giữ track 6 keyframe | 4145 / 134 / 123 | 3765 | 23853 / 972 / 593 | 0.938 | 21695 |
| OCR, sống 4 keyframe | 4129 / 137 / 129 | 3734 | 23872 / 980 / 576 | 0.939 | 21740 |
| OCR + ORU | 4128 / 137 / 129 | 3733 | 23898 / 979 / 569 | 0.939 | 21781 |
| chỉ ORU | 4074 / 131 / 106 | 3731 | 23382 / 949 / 471 | 0.943 | 21491 |
| chỉ OCM | 4074 / 131 / 106 | 3731 | 23321 / 948 / 493 | 0.942 | 21387 |

**Áp dụng: giữ track 4 keyframe** (`propagation3d.max_misses: 2 → 4`). Trên dev đây là mức tốt nhất (6 keyframe ngang
điểm nhưng thêm đổi ID). Test xác nhận: nhãn đúng 23355 → 23731 (+1.6%), vật được phủ 64.3% → 65.3%, đổi ID 470 → 543.

Bước OCR (nhận lại theo vị trí quan sát cuối) thêm được trên test nhưng kém hơn trên dev, nên để tắt (bật được trên tab
Cài đặt). ORU không đổi gì vì mô hình 3D đã dự đoán vận tốc cho từng box.

## Kết luận

Ý chính của OC-SORT — **đừng bỏ track quá sớm khi vật bị che** — có ích cho 3D (+1.6% nhãn đúng). Các thành phần còn lại
không thêm được gì, vì tracker của sản phẩm đã dự đoán theo quan sát: optical flow ở 2D, vận tốc mô hình và ego pose ở 3D.
Chênh lệch lớn giữa OC-SORT và ByteTrack trong bài báo đến từ DanceTrack (người nhảy, chuyển động phi tuyến). Trên MOT17
(người đi bộ, gần tuyến tính) hai phương pháp gần như ngang nhau (HOTA 63.2 và 63.1); xe trên đường giống trường hợp sau.
