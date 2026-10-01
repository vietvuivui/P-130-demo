# Thử ý tưởng VESPA cho 3D

VESPA (arXiv:2507.20397, CVPR 2026 Findings) sinh nhãn 3D không cần nhãn người: Grounding DINO + SAM tìm vật trên ảnh,
chiếu LiDAR vào mask, lọc cụm điểm, nối vật qua các frame, chỉnh hướng theo chuyển động và kích thước theo cỡ lớp.
Sản phẩm đã có mô hình 3D học có nhãn trên nuScenes (held-out mAP 0.668, `det3d_heldout.md`), nên câu hỏi là: các bước
của VESPA có **bổ sung** được gì cho ensemble hiện tại không.

Giao thức: chọn trên dev (3 scene, 119 keyframe), xác nhận trên held-out (24 scene, 957 keyframe), so với nhãn gốc
nuScenes. Số liệu thô: `vespa.json`.

## 1. Box 3D đề xuất từ box 2D + điểm LiDAR (vật mô hình 3D bỏ sót)

`src/services/fill3d.py`: với mỗi box 2D của YOLOE trên 6 camera (có mask) không trùng hình chiếu box 3D nào, lấy điểm
LiDAR (gộp 5 lần quét, bỏ mặt đường) trong mask, gom cụm (DBSCAN), dựng box theo cụm + cỡ trung bình lớp, đặt đáy lên
mặt đường. Điểm = điểm 2D × hệ số, để xếp sau box của mô hình 3D.

Dev, ensemble mặc định; "tìm thấy" / "box sai" đếm ở ngưỡng duyệt 0.3:

| Cấu hình | mAP | NDS | Vật tìm thấy | Box sai | Đề xuất (đúng) |
|---|---|---|---|---|---|
| Ensemble (hiện tại) | **0.644** | **0.643** | 2984 | 1135 | — |
| + đề xuất mọi lớp, điểm × 0.6 | 0.595 | 0.617 | 3053 (+69) | 1361 (+226) | 450 (105 = 23%) |
| + đề xuất mọi lớp, điểm × 0.3 | 0.638 | 0.641 | 2984 | 1135 | 450 (105) |
| + chỉ người / cọc tiêu / xe đạp / xe máy, điểm × 0.6 | 0.627 | 0.634 | 3038 (+54) | 1164 (+29) | 83 (57 = 69%) |
| + chỉ lớp nhỏ, điểm × 0.3 | 0.640 | 0.642 | 2984 | 1135 | 83 (57) |

**Không tăng mAP ở cấu hình nào.** Lý do đo được:

- Vật ensemble bỏ sót (trong tầm, 450 vật trên dev) hầu như không có điểm LiDAR: 63% có dưới 5 điểm, trung vị 3 điểm.
  VESPA dựng box từ cụm điểm, nên không có gì để dựng.
- Đề xuất cho xe / barrier / xe tải gần như luôn sai (~5% đúng): cụm điểm là xe đã có box lệch, tường, bụi cây.
- Người, cọc tiêu, xe đạp đúng 50–75%, nhưng phần lớn trùng vật ensemble đã thấy với điểm thấp (0.1–0.3). Đẩy chúng lên
  làm xáo thứ hạng, AP xe đạp giảm (0.744 → 0.575).

Với người duyệt: bản chỉ lớp nhỏ thêm ~1.8% vật tìm thấy ở ngưỡng duyệt, đổi lấy 29 box sai phải xoá trên 119 keyframe.
Lợi ích nhỏ và chưa xác nhận trên held-out (cần detect 6 camera cho 24 scene trên GPU), nên **không đưa vào sản phẩm**;
`fill3d.py` giữ lại để thử tiếp.

## 2. Hướng theo chuyển động, kích thước theo cỡ lớp

Sau ensemble + track: vật đang chạy (vận tốc track > ngưỡng) thì lật hướng cho cùng chiều chuyển động, hoặc gán hẳn
hướng = hướng vận tốc; kích thước không nhỏ hơn 80–90% cỡ trung bình lớp.

| Cấu hình | mAP dev | NDS dev | mAP held-out | NDS held-out |
|---|---|---|---|---|
| Ensemble (hiện tại) | 0.644 | **0.643** | 0.668 | **0.713** |
| + lật hướng theo chuyển động (v > 1 m/s) | 0.644 | 0.642 | 0.668 | 0.713 |
| + hướng = hướng vận tốc (v > 3 m/s) | 0.644 | 0.642 | 0.668 | 0.713 |
| + cỡ ≥ 80% cỡ lớp | 0.644 | 0.643 | 0.668 | 0.712 |

Không tăng: hướng của mô hình (AOE 0.22) đã tốt hơn hướng suy từ vận tốc track (nhiễu khi xe chạy chậm); kích thước
mô hình học từ nhãn nuScenes nên nới theo cỡ lớp chỉ làm sai số kích thước tăng. Bước lật hướng theo đa số track
(`refine3d`) đã làm phần có ích.

## Kết luận

Các bước của VESPA giải bài toán **không có nhãn 3D**. Khi đã có mô hình học có nhãn trên đúng miền dữ liệu, chúng không
làm tăng độ chính xác. Phần VESPA cho thấy quan trọng nhất (bỏ tracking: −21% mAP) sản phẩm đã có:
tinh chỉnh theo track cho +0.039 mAP (CenterPoint voxel), và lan truyền 3D theo track.
