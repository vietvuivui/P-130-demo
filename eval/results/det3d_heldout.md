# Mô hình 3D trên tập test (held-out) so với nhãn gốc nuScenes

27 scene val của nuScenes trainval có đủ dữ liệu trên máy (1076 keyframe), tách theo giao thức chống overfit:

- **dev** = 3 scene demo của UI (scene-0035/0097/0101, 119 keyframe): dùng để chọn cách làm;
- **held-out** = 24 scene còn lại (957 keyframe): chỉ để báo cáo, không dùng để chọn gì.

Số chuẩn nuScenes detection (mAP theo khoảng cách tâm, NDS), so với nhãn gốc của dataset. Dự đoán lấy từ
`tools3d/work/preds/` (trọng số MMDetection3D có sẵn, không TTA). Ensemble = `refine3d.fuse` + `refine_tracks`, đúng như
bước `run3d.py eval` / `predict` của sản phẩm. Số liệu thô: `eval/results/det3d/heldout24.json`.

| Mô hình | mAP dev | NDS dev | **mAP held-out** | **NDS held-out** | ATE | AOE | AVE |
|---|---|---|---|---|---|---|---|
| PointPillars | 0.394 | 0.503 | 0.473 | 0.559 | 0.39 | 0.56 | 0.30 |
| SSN | 0.451 | 0.510 | 0.441 | 0.568 | 0.33 | 0.35 | 0.33 |
| CenterPoint pillar | 0.431 | 0.522 | 0.528 | 0.610 | 0.29 | 0.37 | 0.36 |
| CenterPoint voxel | 0.537 | 0.560 | 0.578 | 0.655 | 0.26 | 0.29 | 0.37 |
| FCOS3D (camera) | 0.281 | 0.330 | 0.336 | 0.421 | 0.68 | 0.45 | 1.73 |
| PGD (camera) | 0.323 | 0.355 | 0.399 | 0.458 | 0.60 | 0.38 | 1.29 |
| CenterPoint voxel + track | 0.573 | 0.583 | 0.616 | 0.689 | 0.26 | 0.21 | 0.33 |
| Gộp 4 LiDAR | 0.617 | 0.621 | 0.637 | 0.690 | 0.23 | 0.27 | 0.29 |
| **Gộp 4 LiDAR + track (mặc định)** | **0.644** | **0.643** | **0.668** | **0.713** | 0.23 | 0.22 | 0.27 |
| Gộp 4 LiDAR + 2 camera + track | 0.639 | 0.640 | 0.662 | 0.711 | 0.24 | 0.21 | 0.28 |

AP từng lớp của cấu hình mặc định trên held-out: car 0.886 · pedestrian 0.900 · bus 0.803 · traffic_cone 0.802 ·
trailer 0.620 · truck 0.587 · bicycle 0.561 · motorcycle 0.555 · barrier 0.495 · construction_vehicle 0.473.

## Vì sao chọn cấu hình này

- **CenterPoint voxel** là mô hình đơn tốt nhất (held-out 0.578 mAP). Mô hình camera kém xa (0.34–0.40) và sai vận tốc
  gấp 4–6 lần.
- **Tinh chỉnh theo track** (kích thước theo trung vị track, vận tốc theo sai phân vị trí, lật hướng theo đa số, bù
  keyframe bị hụt): +0.039 mAP cho CenterPoint voxel. Gán nhãn được nhìn cả tương lai, nên dùng được track hai chiều.
- **Gộp 4 mô hình LiDAR**: +0.059 mAP so với CenterPoint voxel; cộng track: **+0.090** (0.578 → 0.668).
- Thêm 2 mô hình camera làm ensemble **kém đi** (0.668 → 0.662): đã thấy trên dev, held-out xác nhận.
- Lựa chọn làm trên dev (0.537 → 0.644); held-out tăng cùng chiều và cùng cỡ (0.578 → 0.668), nên không phải overfit dev.
