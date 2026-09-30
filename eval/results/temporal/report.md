# Optical flow cho lan truyền nhãn và QA temporal — trước / sau

Ý tưởng từ *Deep Feature Flow for Video Recognition* (Zhu et al., arXiv:1611.07715) và *Flow-Guided Feature
Aggregation*: tính kỹ ở keyframe, dùng optical flow để mang kết quả sang ảnh lân cận. Hai bài làm ở mức feature map
(phải train lại detector + FlowNet); ở đây làm ở mức box (`src/services/flow.py`, OpenCV DIS), không đổi detector.

- **Idea 1** — lan truyền nhãn: tracker dự đoán box ở ảnh 12Hz kế tiếp bằng flow thay vì vận tốc không đổi.
- **Idea 2** — QA temporal: dời box của sweep t−2…t+2 về thời điểm keyframe bằng flow trước khi so khớp, và tính lại
  score keyframe theo các sweep (`mean` / `linked`).

## Dữ liệu và cách đo

- nuScenes v1.0-trainval, CAM_FRONT, YOLOE-26L (cấu hình mặc định), LiDAR keyframe cho QA.
- **Dev**: scene-0035, 0097, 0101 (3 scene demo của UI, 119 keyframe). **Held-out**: scene-0003, 0016, 0039, 0095
  (scene val có đủ ảnh trên máy, chọn đều 1/5 theo thứ tự tên — chọn trước khi chạy, 159 keyframe).
- Mọi cấu hình dùng **cùng một detection** (cache của lần `run`), chỉ khác phần xử lý sau detector.
  `tools2d/eval_temporal.py`; số liệu đầy đủ trong `dev.json`, `heldout.json`.
- GT 2D = hộp bao box 3D chiếu xuống ảnh (rộng hơn hộp detector), khớp khi IoU ≥ 0.5 cùng lớp.
- Cấu hình mặc định được chọn **trên dev**, trước khi chạy held-out.

## 0. Held-out lớn: 20 scene val, 795 keyframe (laptop GPU, `scripts\tasks.ps1 evaltemporal`)

Chạy lại trên GPU của Kiên với toàn bộ 20 scene val có đủ dữ liệu (khác 3 scene dev). Số liệu: `gpu_heldout20.json`
(dev chạy lại trên GPU: `gpu_dev.json`, khớp số trên CPU trong sai số fp16).

**Lan truyền** (154 lần, 1060 object):

| `propagation.flow` | Nhãn đúng / box ra | Đổi ID | Mất dấu | Box sai | IoU TB | Giây (tổng) |
|---|---|---|---|---|---|---|
| off (trước) | 2910 / 4531 (64.2%) | 265 | 1150 | 1356 | 0.508 | 51 |
| missing | 3068 / 4484 (68.4%) | 181 | 1055 | 1235 | 0.541 | 51 |
| **always** (mặc định) | **3329 / 4693 (70.9%)** | **97** | **902** | 1267 | **0.567** | 214 |

**2D:**

| Cấu hình | mAP50 | F1 | Xem tay | Lọt qua | Vẽ thêm | Box sai | Lỗi còn lại (lọt + sót) |
|---|---|---|---|---|---|---|---|
| base (mặc định) | 0.295 | 0.504 | 1707 | 1632 | 2565 | 2980 | 4197 |
| flow | 0.295 | 0.504 | 1059 | 2082 | 2565 | 2980 | 4647 |
| flow + mean | 0.291 | 0.523 | 492 | 1906 | 2660 | 2300 | 4566 |
| flow + linked | 0.304 | 0.495 | 1415 | 2319 | 2444 | 3544 | 4763 |
| **mean (không flow)** | 0.283 | **0.527** | 648 | **1399** | 2791 | **1860** | **4190** |

- Idea 1 được xác nhận trên dữ liệu gấp 5 lần: +14% nhãn lan truyền đúng, đổi ID −63%, mất dấu −22%, box sai −7%.
- Idea 2: flow cho QA temporal làm lỗi lọt qua tăng rõ — không khuyên bật. `mean` không flow giữ tổng lỗi như cũ,
  bớt 62% box phải xem và 38% box sai phải xoá, đổi lại thêm 9% vật sót phải vẽ. Mặc định vẫn tắt (đã chọn trên dev);
  nhóm có thể bật ở tab ⚙ Cài đặt nếu muốn đổi việc xoá / xem lấy việc vẽ thêm.

## 1. Box sweep khớp keyframe (flow có dời đúng không)

Box detector ở sweep so với box gần nhất ở keyframe:

| Sweep | IoU TB không flow → có flow (dev) | Tỉ lệ IoU ≥ 0.5 (dev) | IoU TB (held-out) | Tỉ lệ IoU ≥ 0.5 (held-out) |
|---|---|---|---|---|
| t−2 | 0.585 → **0.828** | 0.61 → **0.92** | 0.641 → **0.832** | 0.69 → **0.93** |
| t−1 | 0.723 → **0.881** | 0.84 → **0.97** | 0.770 → **0.890** | 0.88 → **0.97** |
| t+1 | 0.702 → **0.877** | 0.79 → **0.96** | 0.750 → **0.885** | 0.86 → **0.97** |
| t+2 | 0.581 → **0.832** | 0.60 → **0.93** | 0.634 → **0.829** | 0.68 → **0.93** |

## 2. Idea 1 — lan truyền nhãn (thí nghiệm keyframe hoàn hảo)

GT ở keyframe 0, 5, 10… của mỗi scene đóng vai nhãn người đã duyệt; lan truyền tối đa 10 keyframe (5 s) qua mọi ảnh
12Hz; so với GT cùng `instance_token`. Cache detection chỉ có keyframe và t±1, t±2 như khi dùng thật (ảnh giữa hai
keyframe không có detection).

| `propagation.flow` | Dev: đúng / box ra | Đổi ID | Mất dấu | IoU TB | Held-out: đúng / box ra | Đổi ID | Mất dấu | IoU TB |
|---|---|---|---|---|---|---|---|---|
| off (trước) | 725 / 1041 (69.6%) | 49 | 273 | 0.549 | 785 / 1106 (71.0%) | 26 | 237 | 0.580 |
| missing | 741 / 1003 (73.9%) | 38 | 281 | 0.580 | 851 / 1154 (73.7%) | 19 | 238 | 0.593 |
| **always** (mặc định) | **791 / 1038 (76.2%)** | **23** | **251** | **0.599** | **873 / 1213 (72.0%)** | **11** | **183** | 0.584 |

- Held-out: `always` cho **thêm 88 nhãn lan truyền đúng (+11%)**, **đổi ID giảm 58%** (26 → 11), mất dấu giảm 23%;
  đổi lại giữ được nhiều track hơn nên số box sai tăng nhẹ (295 → 329). `missing` thận trọng hơn: +66 nhãn đúng, sai
  giảm (295 → 284).
- c_prop của nhãn sai tăng (dev 0.59 → 0.71) nên cờ `PROP_LOW_CONF` bắt được ít lỗi hơn (recall dev 0.42 → 0.24);
  tổng số lỗi không bị gắn cờ gần như không đổi (dev 182 → 188) vì số lỗi giảm.
- Chi phí: ~0.05–0.1 s mỗi ảnh 12Hz trên CPU 2 nhân (một lần lan truyền 10 keyframe ≈ 3 s).

## 3. Idea 2 — QA temporal và score theo sweep

| Cấu hình | mAP50 dev | F1 dev | Box phải xem tay | Lỗi lọt qua duyệt theo lô | Phải vẽ thêm (FN) | Box sai (FP) | mAP50 held-out | F1 held-out | Xem tay | Lọt qua | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (trước, mặc định) | 0.377 | 0.574 | 304 | **252** | 452 | 475 | 0.359 | 0.603 | 240 | **277** | 385 | 485 |
| flow (chỉ so khớp) | 0.377 | 0.574 | 219 | 306 | 452 | 475 | 0.359 | 0.603 | 172 | 327 | 385 | 485 |
| flow + mean | 0.369 | **0.594** | **106** | 279 | 474 | **349** | 0.358 | 0.626 | **96** | 305 | 392 | 388 |
| flow + linked | 0.378 | 0.562 | 297 | 354 | **423** | 593 | **0.365** | 0.589 | 225 | 352 | **375** | 561 |
| mean (không flow) | 0.365 | 0.589 | 121 | 219 | 505 | 293 | 0.338 | 0.629 | 102 | 248 | 411 | 337 |

"Box phải xem tay" = box rủi ro medium/high; "lọt qua" = box sai nằm trong nhóm low (duyệt theo lô không xem).

- **mAP gần như không đổi** (±0.01), kể cả khi giữ mọi box ≥ 0.10 để so thứ hạng (dev 0.395 → 0.397 / 0.391 / 0.403;
  held-out 0.368 → 0.368 / 0.371 / 0.354). Tính lại score chủ yếu dời điểm làm việc (precision ↔ recall), không làm
  detector xếp hạng tốt hơn.
- `flow + mean`: **bớt 60–65% box phải xem tay** và ~100 box sai, nhưng **lỗi còn lại sau duyệt tăng 5–7%**
  (dev 704 → 753, held-out 662 → 697): một phần box sai xuất hiện đều ở cả 5 ảnh (vật ngoài taxonomy, box lệch GT),
  trước đây bị cờ FLICKER "bắt nhầm" vì vật đang chạy làm box sweep lệch, nay khớp được nên không còn cờ.
- Flow làm FLICKER đúng hơn (precision held-out 0.84 → 0.94) nhưng ít cờ hơn (160 → 82).

**Mặc định:** `qa.temporal.flow: false`, `qa.temporal.rescore: off` — ưu tiên chất lượng nhãn cuối cùng. Nếu nhóm
chấp nhận thêm ~5% lỗi để bớt ~60% việc xem tay, bật `flow: true` + `rescore: mean`.

## 4. Thời gian (CPU 2 nhân, không GPU)

| Bước | s / keyframe |
|---|---|
| Detect 5 ảnh (keyframe + 4 sweep, YOLOE-L 1280) | 17.4–17.9 |
| QA không flow (detection đã cache) | 0.01 |
| QA có flow (4 cặp ảnh) | 0.2–0.3 (+1.5% so với detect) |

## Sửa nhãn ở sweep

Box ở t−2…t+2 giờ sửa được trên UI (keep / xoá / đổi lớp / sửa box / vẽ thêm); keyframe được tính lại ngay và tracker
lan truyền dùng bản đã sửa. Không nằm trong số liệu trên (cần người duyệt thật); kiểm tra bằng
`tests/test_api/test_sweep_edit.py`: vẽ vật detector sót ở t−1, t+1 → keyframe có đề xuất `RECOVERED_BY_TRACK`;
thêm vật ở sweep → object keyframe hết `FLICKER`.
