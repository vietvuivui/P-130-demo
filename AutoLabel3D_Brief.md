# AutoLabel 3D — Công cụ tự động gán nhãn ảnh & LiDAR (2D/3D box, segmentation) với Human-in-the-loop

> Dự án ứng dụng AI trong thực tế — Perception cho xe tự lái (AV)

## Bối cảnh

Gán nhãn dữ liệu perception cho AV (2D box, 3D box, segmentation trên ảnh và point cloud) là công đoạn tốn kém nhất trong phát triển AV — thủ công, chậm, và không đồng nhất giữa các annotator.

## Mục tiêu

Xây dựng công cụ tự động sinh nhãn sơ bộ bằng model pretrained (không train mới), tích hợp giao diện review để annotator xác nhận/chỉnh sửa trước khi vào tập train. Yêu cầu:

- Đồng bộ nhãn giữa ảnh camera và point cloud LiDAR (chiếu 2D↔3D qua ma trận calibration)
- Human-in-the-loop bắt buộc — mọi nhãn tự động phải qua duyệt của con người
- Đo được hiệu quả: mAP/IoU so với ground truth, tỷ lệ nhãn phải sửa, thời gian tiết kiệm
- Tối ưu latency inference & chi phí GPU khi chạy batch lớn
- Ẩn danh khuôn mặt & biển số trước khi hiển thị cho annotator

## Data

**nuScenes v1.0-mini** (3.88GB, 10 scene, ~404 sample) — 6 camera (360°), 1 LiDAR (`x,y,z,intensity`), calibration đầy đủ, ground truth box 3D (23 lớp, 2Hz). Mở rộng segmentation: `nuScenes-lidarseg` (point cloud) hoặc `nuImages` (ảnh 2D).

> Giới hạn: bản mini chỉ 10/1000 scene → số liệu mAP/IoU mang tính minh họa quy trình, chưa đạt độ tin cậy thống kê đầy đủ.

## Model — 3 nhánh song song (inference only, không train)

| Nhánh | Model | Input | Output |
|---|---|---|---|
| 3D Detection | PointPillars (OpenPCDet/MMDetection3D) | Point cloud | Box 3D |
| 2D Detection | YOLOv8 pretrained | Ảnh camera | Box 2D |
| Segmentation *(mở rộng)* | SAM | Ảnh + gợi ý box 2D | Mask pixel-level |

Box 3D chiếu xuống ảnh (qua calibration) đối chiếu với box 2D detect trực tiếp → gắn cờ "cần review" nếu lệch nhiều. Segmentation bổ sung độ chi tiết pixel-level, không đối chiếu chéo với box.

## UX/UI demo

Mở rộng từ **CVAT/Label Studio** (không build UI mới). Hiển thị song song ảnh (đã ẩn danh, có box/mask) và point cloud (có box 3D) cho cùng 1 object. Luồng: `pending review` → annotator sửa/giữ nguyên → `approved` → vào tập train.

**Flow demo:** raw data → auto-label 3 nhánh → đồng bộ & gắn cờ → ẩn danh → annotator duyệt → xuất tập train.

## Đánh giá

| Chỉ số | Mục đích |
|---|---|
| mAP / IoU | Độ chính xác so với ground truth |
| Tỷ lệ nhãn phải sửa | Độ tin cậy của nhãn tự động |
| Thời gian tiết kiệm | So với gán nhãn thủ công hoàn toàn |
| Latency & chi phí GPU | Hiệu năng khi chạy batch lớn |

## Phạm vi

Dừng ở công cụ gán nhãn + tập dữ liệu đã xác nhận — **không** train lại model detection. Fine-tune model bằng data đã duyệt (vòng lặp cải thiện) là hướng mở rộng, không bắt buộc.
