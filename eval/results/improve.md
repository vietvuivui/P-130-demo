# Bốn cải tiến cho vật nhỏ / thấy mờ — đo ngày 03/10/2026

Xuất phát từ hai lỗi thấy khi test demo: xe đạp có nhãn (0.34) ở frame này rồi biến mất ở frame sau (detector chỉ còn
0.20, dưới ngưỡng giữ 0.30), và ô tô ở xa được detector thấy 0.58 nhưng bị hạ còn 0.29 khi gộp với 3D (không có box 3D)
rồi bị bỏ. Bốn ý tưởng sửa gốc được cài thành cấu hình và đo bằng `scripts\tasks.ps1 improve` (RTX 4050, detector
YOLOE-26-L fine-tune nuImages; số liệu thô: `improve/*.json`). Chọn trên dev (3 scene), báo trên held-out / test.

| # | Ý tưởng | Kết luận | Mặc định |
|---|---|---|---|
| 1 | Detector nhìn vật nhỏ tốt hơn: chạy ở 1920, hoặc thêm lưới ô 2×2 | 1920 **kém hơn** (model học ở 1280). Lưới ô bắt thêm 4% vật (vật < 32 px: 37% → 46%) nhưng box sai tăng 60%, việc phải sửa tăng 32% | tắt |
| 2 | Không hạ điểm box chỉ camera thấy khi trong box không có điểm LiDAR | Sót giảm 8% nhưng box sai **tăng gấp đôi**: phần lớn box kiểu này là box sai thật | giữ hạ 0.5 |
| 3 | Giữ box detector thấy mờ khi sweep trước và sau đều thấy (RECOVERED_BY_TRACK) | Bù được thêm 73 vật sót trên held-out (gấp 3 lần trước), nhưng chỉ 11% đề xuất trúng vật; mỗi frame thêm ~0.8 box nét đứt phải bấm bỏ | bật (0.2), xem ghi chú |
| 4 | Mang nhãn frame trước bằng tracker lúc gán nhãn | Thêm 28 vật bù được nữa, cùng tỉ lệ trúng 11%, tốn thêm 0.26 s/frame (optical flow) | tắt |

## 1. Detector: độ phân giải và lưới ô (`tools2d/eval2d.py --imgsz / --tiles`)

Held-out 957 keyframe, ngưỡng giữ 0.30:

| Cấu hình | mAP50 | P | R | F1 | Recall vật < 32 px | 32–64 px | 64–128 px | ≥ 128 px |
|---|---|---|---|---|---|---|---|---|
| 1280 (hiện tại) | 0.366 | 0.484 | 0.581 | 0.528 | 98/268 (37%) | 999/1906 (52%) | 1622/2570 (63%) | 1323/1989 (67%) |
| 1920 | 0.344 | 0.458 | 0.573 | 0.509 | 106/268 (40%) | 1074/1906 (56%) | 1641/2570 (64%) | 1206/1989 (61%) |
| 1280 + lưới ô 2×2 | 0.375 | 0.385 | 0.620 | 0.475 | 122/268 (46%) | 1148/1906 (60%) | 1719/2570 (67%) | 1341/1989 (67%) |

Dev (119 keyframe): mAP50 0.429 / 0.370 / 0.427; F1 0.574 / 0.535 / 0.492.

- **1920:** model fine-tune ở 1280 nên chạy ở 1920 làm vật lớn kém đi (≥ 128 px: 67% → 61%) nhiều hơn phần được ở vật
  nhỏ. Muốn dùng 1920 phải fine-tune lại ở 1920.
- **Lưới ô:** đúng là bắt thêm vật nhỏ (+24 vật < 32 px, +149 vật 32–64 px trên held-out) và mAP50 tăng nhẹ, nhưng mỗi
  ô nhìn một mảnh ảnh thiếu ngữ cảnh nên sinh nhiều box sai: ước tính box sai 4170 → 6670 (+60%), việc phải sửa
  (thừa + sót) 6990 → 9230 (+32%). Chưa thử: lưới ô kết hợp gộp 3D (bước gộp hạ điểm box chỉ camera thấy, có thể lọc
  bớt box sai của ô) — cần đo trước khi dùng.

## 2. Gộp box 3D: hạ điểm box chỉ camera thấy (`tools2d/eval_lidar2d.py`)

Test 24 scene, 957 keyframe, 6733 vật; ngưỡng giữ 0.30:

| Cấu hình | mAP50 | P | R | F1 | Đúng | Thừa (xoá) | Sót (vẽ) | Phải sửa |
|---|---|---|---|---|---|---|---|---|
| Chỉ detector ảnh | 0.366 | 0.484 | 0.580 | 0.528 | 3907 | 4164 | 2826 | 6990 |
| **Gộp, hạ ×0.5 mọi box chỉ camera thấy (hiện tại)** | **0.628** | **0.717** | 0.733 | **0.725** | 4934 | **1949** | 1799 | **3748** |
| Gộp, không hạ khi box không có điểm LiDAR | 0.597 | 0.562 | **0.753** | 0.644 | 5071 | 3955 | 1662 | 5617 |
| Gộp, hạ ×0.7 khi box không có điểm LiDAR | 0.621 | 0.644 | 0.745 | 0.691 | 5019 | 2774 | 1714 | 4488 |

Dev: phải sửa 637 / 935 / 748 (cùng thứ tự ba dòng gộp).

Giữ nguyên quy tắc hiện tại. Box chỉ camera thấy mà không có điểm LiDAR gồm cả vật ở xa thật (như ô tô ở scene-0101_024)
lẫn box sai (vật ngoài taxonomy, bóng, biển hiệu); số sau nhiều hơn hẳn: tha cho nhóm này thì cứ cứu được 1 vật sót
lại thêm 15 box sai. Trường hợp ô tô ở xa đã được xử lý ở bước QA: box bị bỏ trùng detection keyframe thì được đề xuất
lại bằng chính box của detector, nét đứt, ghi rõ "score 0.58, còn 0.29 sau khi gộp với 3D" (cải tiến 3).

## 3–4. QA temporal: giữ box thấy mờ, mang nhãn frame trước (`tools2d/eval_temporal.py`)

Cùng detection, chỉ khác xử lý sau detector. "Đề xuất" = box RECOVERED_BY_TRACK (nét đứt, người phải xác nhận hoặc bấm
bỏ, không bao giờ vào nhóm duyệt theo lô); "đúng" = trùng một vật thật; "bù sót" = trùng vật mà detector sót (người đỡ
phải vẽ). mAP50 / F1 / xem tay / lọt lô / box sai **không đổi** giữa các cấu hình vì đề xuất không tính vào box của model.

Held-out 795 keyframe (vật sót 2243):

| Cấu hình | Đề xuất | Đúng | Bù sót | Tỉ lệ trúng | Đề xuất sai | s/frame |
|---|---|---|---|---|---|---|
| noweak (trước 02/10) | 329 | 35 | 35 | 10.6% | 294 | 0.32 |
| **weak** (giữ box thấy mờ, mặc định) | 943 | 108 | 108 | 11.5% | 835 | 0.28 |
| carry (weak + mang nhãn frame trước) | 1200 | 136 | 136 | 11.3% | 1064 | 0.54 |
| carry-only | 741 | 94 | 94 | 12.7% | 647 | 0.54 |

Dev 119 keyframe (vật sót 395): đề xuất 58 / 225 / 276 / 156, đúng 11 / 32 / 37 / 25.

- **Cải tiến 3 (weak)** bù được gấp 3 lần vật sót so với trước (35 → 108, bằng 4.8% số vật sót), không tốn thêm thời
  gian. Nhưng 88% đề xuất là sai: trên held-out, mỗi frame có thêm ~0.8 box nét đứt phải bấm bỏ để đổi lấy 0.09 vật khỏi
  phải vẽ. Có đáng hay không tuỳ chi phí thao tác: bỏ một đề xuất là một phím (`D`), vẽ một box là kéo chuột + chọn
  lớp; với ước lượng 1 s / 6 s thì hai bên gần hoà. Giữ bật vì sửa đúng lỗi người dùng gặp (xe đạp biến mất giữa hai
  frame) và đề xuất sai không lọt vào nhãn cuối; tắt bằng `qa.temporal.recover_weak_min_score: null`.
- **Cải tiến 4 (carry)** thêm 28 vật bù được với cùng tỉ lệ trúng, nhưng tốn optical flow ~0.26 s/frame. Không bật mặc
  định; bật bằng `qa.temporal.carry_prev: true` khi cần recall tối đa (video tải lên không có LiDAR).
- Ghi chú: tỉ lệ trúng có thể bị đánh giá thấp vì vật khuất > 60% / không có điểm LiDAR bị bỏ khỏi nhãn gốc khi chấm,
  mà đề xuất hay rơi đúng vào vật kiểu đó. Chưa đo riêng.

## Việc còn lại

- Fine-tune lại YOLOE ở 1920 (hoặc đa tỉ lệ) rồi đo lại mục 1 — cách duy nhất trong bốn ý tưởng còn hy vọng nâng detector
  mà không tăng box sai.
- Lưới ô + gộp 3D (chưa đo).
- Chỉnh `recover_weak_min_score` (0.2 → 0.25 / 0.3) và `match_iou` của đề xuất trên dev để nâng tỉ lệ trúng.

Chạy lại: `scripts\tasks.ps1 improve` (GPU ~10 phút cho detector, CPU ~25 phút cho phần còn lại).
