# Dò lại QA Agent 2D cho detector fine-tune toàn mạng (2026-10-04)

Sau khi đổi detector 2D sang bản fine-tune toàn mạng, QA Agent 2D bắt lỗi kém hẳn: 82% box rơi vào nhóm low (duyệt theo
lô) mà 48% số đó sai. Tài liệu này ghi lại nguyên nhân, cấu hình mới và số đo trước / sau.

- **Dữ liệu:** nuScenes val, CAM_FRONT. Chọn cấu hình trên **dev** (3 scene, 119 keyframe, 1883 box); báo trên
  **held-out** (20 scene, 795 keyframe, 9241 box). "Box sai" = box không khớp nhãn gốc cùng lớp ở IoU ≥ 0.5.
- **Lệnh:** `python tools2d/eval_temporal.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_heldout_full
  --out <thư mục> --variants base qa-old --no-propagation` (`base` = cấu hình mới, `qa-old` = cấu hình trước 04/10).
- **Số liệu:** `eval/results/qa/qa_dev.json`, `qa_heldout20.json`.

## Kết quả (held-out)

| Chỉ số | QA cũ | QA mới |
|---|---|---|
| mAP50 / P / R / F1 của nhãn máy | 0.611 / 0.475 / 0.815 / 0.600 | không đổi |
| Nhóm low (duyệt theo lô): số box | 7573 (82%) | 3138 (34%) |
| Nhóm low: tỉ lệ sai | 47.8% | **8.9%** |
| Box sai lọt qua duyệt theo lô | 3621 | **280** |
| Box phải xem tay (medium + high) | 1668 | 6103 |
| Tỉ lệ box sai được gắn cờ | 25% | **94%** |
| Nhóm high: số box / tỉ lệ sai | 13 / 92% | 2300 / **95%** |
| Đề xuất RECOVERED_BY_TRACK / số trùng vật bị sót | 627 / 37 | 0 / 0 |

Dev: nhóm low 600 box (32%), sai 5.2%; nhóm high 480 box, sai 94.8%; box sai lọt 780 → 31.

QA Agent **không làm đổi mAP, precision, recall, F1**: các chỉ số đó chỉ tính trên box của detector. QA chỉ quyết
định box nào người phải xem và box nào được duyệt theo lô.

Cái giá của cấu hình mới: người duyệt phải xem 66% số box thay vì 18%. Lý do là detector hiện ra 52% box sai so với
nhãn gốc; với tỉ lệ đó không có cách xếp hạng nào cho nhóm duyệt theo lô vừa lớn vừa sạch. Nhóm high (25% số box, 95%
sai) là ứng viên cho thao tác "xoá theo lô", hiện giao diện chưa có.

## Vì sao QA cũ kém

Đo khả năng xếp box sai lên trên box đúng (AUC, 1.0 = hoàn hảo, 0.5 = ngẫu nhiên):

| Cách chấm | Dev | Held-out |
|---|---|---|
| Risk cũ của QA Agent | 0.847 | 0.841 |
| Chỉ dùng score detector (không có QA Agent) | 0.860 | 0.841 |
| Điểm LiDAR cũ (riêng) | 0.776 | 0.787 |
| Điểm temporal (riêng) | 0.569 | 0.584 |
| Điểm hình học (riêng) | 0.500 | 0.500 |
| **Risk mới** | **0.924** | **0.916** |
| Hồi quy logistic trên score + LiDAR (tham khảo, chặn trên) | 0.928 | 0.917 |

1. **Risk cũ không hơn gì score detector.** Trọng số và ngưỡng được chọn cho detector zero-shot.
2. **Ngưỡng lệch với score của detector mới.** Tỉ lệ sai theo score (held-out): 0.3–0.6 sai 85%, 0.6–0.7 sai 71%,
   0.7–0.8 sai 48%, 0.8–0.9 sai 29%, từ 0.9 sai 2%. Với công thức cũ, box score 0.5 không có cờ nào có risk 0.175,
   tức vẫn là low.
3. **Tín hiệu LiDAR bị dùng quá nhẹ.** Tỉ lệ sai theo số điểm LiDAR trong box (held-out): 0 điểm 96% (2231 box),
   1 điểm 79%, 2 điểm 68%, 3–4 điểm 51%, 5–9 điểm 32%, từ 10 điểm khoảng 20%. Cấu hình cũ chỉ gắn cờ khi box cao
   ≥ 60 px; box nhỏ không có điểm nào chỉ bị cộng 0.2. Phần lớn box loại này là vật ở xa mà nhãn gốc nuScenes không
   gán (nhãn gốc chỉ gán vật có điểm đo).
4. **Điểm temporal gần như không giúp xếp hạng** (AUC 0.58). Cờ FLICKER vẫn có ích (69% box bị cờ này là sai).
5. **Đề xuất RECOVERED_BY_TRACK gần như toàn sai:** 314 box nội suy chỉ 7 trùng vật bị sót (kể cả khi hai sweep có
   score ≥ 0.7: 2 / 134), 313 box "thấy mờ" chỉ 31 trùng. Người duyệt phải gạt khoảng 590 đề xuất để bù 37 vật.

## Cấu hình mới (`configs/autolabel.yaml`)

| Tham số | Trước | Sau |
|---|---|---|
| `qa.risk.weights` (detection / lidar / temporal / geometric) | 0.35 / 0.30 / 0.20 / 0.15 | 0.60 / 0.25 / 0 / 0.15 |
| `qa.risk.levels` (medium / high) | 0.30 / 0.60 | 0.17 / 0.40 |
| `qa.risk.issue_floor` | 0.30 | 0.17 |
| `qa.confidence.low_threshold` | 0.35 | 0.60 |
| Box nhỏ, 0 điểm LiDAR | cộng 0.2, không cờ | `no_points_term` 1.0 + cờ `NO_LIDAR_SUPPORT` |
| Box nhỏ, 1–2 điểm | cộng 0.2 | `few_points_term` 0.4 |
| Box có 3–4 điểm | 0 | `sparse_term` 0.2 (`sparse_points` 5) |
| `qa.temporal.recover_min_score` | 0.35 | 1.0 (chỉ box sweep người đã xác nhận / vẽ mới tạo đề xuất) |
| `qa.temporal.recover_weak_min_score` | 0.2 | null (tắt) |

Cách chọn, đều trên dev:

- **Trọng số và điểm LiDAR:** dò lưới, lấy bộ có AUC cao nhất.
- **Ngưỡng medium:** ngưỡng lớn nhất mà nhóm low sai ≤ 5%. Trên held-out thành 8.9%, dưới mức 10% mà bước audit duyệt
  theo lô yêu cầu (`qc.audit.object_max_error_upper`).
- **Ngưỡng high:** ngưỡng nhỏ nhất mà nhóm high sai ≥ 95%.

## Có nên bỏ QA Agent không

Không nên bỏ hẳn. Nếu chỉ dùng score detector với cùng cách chọn ngưỡng (nhóm low sai ≤ 5% trên dev), held-out có nhóm
low 1564 box (17%), tức người phải xem 7677 box thay vì 6103. Phần hơn đó đến từ kiểm tra LiDAR.

Phần đã lược:

- Điểm temporal ra khỏi công thức risk (trọng số 0).
- Đề xuất RECOVERED_BY_TRACK từ box của riêng detector.

Phần giữ: kiểm tra LiDAR, cờ FLICKER, kiểm tra hình học, và đề xuất RECOVERED_BY_TRACK khi người vẽ / xác nhận box ở
sweep.

## Hai hướng: ngưỡng giữ box thấp (recall cao) hay cao (precision cao)

Thử ngày 04/10/2026. Gán nhãn lại dev và held-out với `detection.min_score` 0.10 (mọi box detector lưu), rồi lọc theo
từng ngưỡng. Ở mỗi ngưỡng, ngưỡng risk được chọn lại trên dev (nhóm low sai ≤ 5%, nhóm high sai ≥ 95%). Số dưới đây là
held-out (795 keyframe, 5384 vật trong nhãn gốc); số liệu: `eval/results/qa/two_approaches.json`. Ở ngưỡng 0.30 mô phỏng
khớp lần chạy thật (xem tay 6092 so với 6103, lọt 284 so với 280).

| Ngưỡng giữ box | Số box | P | R | F1 | Vật sót phải vẽ | Box phải xem tay | Box sai lọt duyệt lô |
|---|---|---|---|---|---|---|---|
| 0.10 (hướng 1) | 11844 | 0.379 | 0.834 | 0.521 | 894 | 8695 | 284 |
| 0.20 | 10207 | 0.435 | 0.826 | 0.570 | 939 | 7058 | 284 |
| **0.30 (hiện tại)** | 9241 | 0.475 | 0.815 | 0.600 | 994 | 6092 | 284 |
| 0.50 | 7487 | 0.562 | 0.782 | 0.654 | 1173 | 4338 | 284 |
| 0.60 | 6399 | 0.622 | 0.739 | 0.675 | 1407 | 3250 | 284 |
| 0.70 (hướng 2) | 5080 | 0.707 | 0.667 | 0.686 | 1793 | 1931 | 284 |
| 0.80 | 3267 | 0.810 | 0.491 | 0.611 | 2739 | 671 | 205 |

Nhóm low (3149 box, sai 9.0%) giống nhau ở mọi ngưỡng tới 0.70, vì nó chỉ gồm box score cao. Đổi ngưỡng giữ box chỉ
đổi một việc: box score thấp được đưa cho người xem (và xoá), hay bị bỏ trước (và người phải vẽ lại những vật đúng nằm
trong đó).

Tỉ lệ box đúng theo dải score (held-out):

| Dải score | Số box | Box đúng |
|---|---|---|
| 0.10–0.30 | 2603 | 100 (3.8%) |
| 0.30–0.50 | 1754 | 179 (10.2%) |
| 0.50–0.60 | 1088 | 234 (21.5%) |
| 0.60–0.70 | 1319 | 386 (29.3%) |
| 0.70–0.80 | 1813 | 946 (52.2%) |
| 0.80–0.90 | 2065 | 1464 (70.9%) |
| từ 0.90 | 1202 | 1181 (98.3%) |

Kết luận:

- **Hướng 1 (hạ ngưỡng xuống 0.10) không đáng:** recall chỉ tăng 0.815 → 0.834 (bớt 100 vật phải vẽ) mà người phải
  xem thêm 2603 box, tức 26 box cho mỗi vật.
- **Hướng 2 (nâng ngưỡng) đáng tới đâu tuỳ công vẽ một box so với công xem một box.** Giữ một dải score có lợi khi tỉ
  lệ box đúng trong dải lớn hơn 1 / (số lần xem tương đương một lần vẽ). Tổng công = box xem tay + hệ số × vật phải vẽ:

| Vẽ một box bằng mấy lần xem | 0.10 | 0.30 | 0.50 | 0.60 | 0.70 | 0.80 | Ngưỡng tốt nhất (dev chọn cùng ngưỡng) |
|---|---|---|---|---|---|---|---|
| 3 | 11377 | 9074 | 7857 | 7471 | **7310** | 8888 | 0.70 |
| 5 | 13165 | 11062 | **10203** | 10285 | 10896 | 14366 | 0.50–0.60 (dev: 0.60) |
| 10 | 17635 | 16032 | 16068 | 17320 | 19861 | 28061 | 0.30–0.50 |

- **Siết QA sau khi nâng ngưỡng** (nhóm low sai ≤ 2% thay vì ≤ 5%): box sai lọt 284 → 28, nhưng nhóm low còn 1344 box
  và người phải xem thêm 1805 box, tức 7 box cho mỗi lỗi bắt thêm.
- Hệ số công vẽ / công xem chưa được đo trên giao diện của nhóm; bảng trên chỉ cho biết ngưỡng nào tốt với từng giả định.
- Vật không có box nào thì người duyệt dễ bỏ qua hơn là box sai hiện trên ảnh. Bảng trên giả định người vẽ lại đủ mọi
  vật sót; nếu không, ngưỡng cao làm nhãn cuối thiếu vật nhiều hơn.
- mAP50 không phụ thuộc lựa chọn này theo cách đáng kể (0.625 ở ngưỡng 0.10, 0.611 ở 0.30).

## QA gọi thêm một detector 2D làm ý kiến thứ hai

Thử ngày 04/10/2026 với hai detector đã có cache trên cùng ảnh: YOLOE-26L zero-shot và bản linear probe (cả hai yếu hơn
detector chính; chưa có model 2D nào mạnh hơn trên miền dữ liệu này để thử). Thêm đặc trưng "detector thứ hai có thấy box
cùng lớp, IoU ≥ 0.5 không, score bao nhiêu" vào hồi quy logistic cùng với score, điểm LiDAR, điểm hình học; học trên dev.

| Cách chấm | AUC dev | AUC held-out |
|---|---|---|
| Score + LiDAR + hình học | 0.881 | 0.878 |
| + YOLOE zero-shot | 0.890 | 0.875 |
| + linear probe | 0.881 | 0.875 |
| + cả hai | 0.890 | 0.875 |

Ý kiến của detector 2D thứ hai không cải thiện việc xếp box sai trên held-out. Detector thứ hai cũng thấy 34–51% số box
"sai" không có điểm LiDAR (so với 74–82% box đúng): phần lớn là vật thật ở xa mà nhãn gốc không gán, nên một model nhìn
ảnh khác cũng đồng ý với detector chính. Giới hạn: phép thử dùng model yếu hơn, không phải mạnh hơn.

## Giới hạn

- Phần QA ở tài liệu này chỉ đo ở dự án có LiDAR và chỉ CAM_FRONT (phần gộp box 3D đã đo 6 camera: `lidar2d.md`). Dự án chỉ có camera không có điểm LiDAR, khi đó risk gần như chỉ còn là
  score detector.
- "Box sai" tính theo nhãn gốc nuScenes. Một phần box 0 điểm LiDAR là vật thật ở xa mà nhãn gốc không gán; với bộ dữ
  liệu muốn gán cả vật xa thì cờ này quá chặt (tắt bằng `qa.lidar.zero_points_issue: false`, `no_points_term: 0.2`).
- Ngưỡng chọn trên 3 scene dev; tỉ lệ sai của nhóm low lệch từ 5% (dev) lên 9% (held-out).
- Chưa đo lại với tuỳ chọn "tính lại score theo sweep" (`rescore`) và optical flow cho QA temporal.
