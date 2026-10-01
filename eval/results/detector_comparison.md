# So sánh detector: YOLO-World vs YOLO26-L / YOLOE-26-L

- Dữ liệu: 79 keyframe CAM_FRONT nuScenes v1.0-mini (scene-0061, scene-0103), keyframe + 4 sweep mỗi frame.
- GT: hộp bao box 3D chiếu xuống ảnh, bỏ object visibility 0–40% (xem `autolabel2d_eval.md`), khớp ở IoU ≥ 0.5.
- Cùng config QA Agent / ngưỡng risk (chỉnh cho YOLO-World); `score_threshold` 0.10, `imgsz` 1280 cho mọi model.
- Fusion chỉ tính phiếu của model nhận được lớp đó (YOLO26 COCO không bị tính là phiếu chống cho cone / barrier).
- Tái lập: `python -m src.cli run --scenes scene-0061 scene-0103 --overwrite --detectors <...>` rồi
  `python -m src.cli evaluate` (detection được cache theo model nên đổi tổ hợp chỉ mất vài giây).

| | YOLO-World | YOLO26-L | YOLOE-26-L (**mặc định**) | 26-L + E26-L | 26-L + World | E26-L + World | 26-L + E26-L + World |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| **mAP@0.5** | 0.320 | 0.236 | 0.281 | 0.298 | 0.312 | 0.315 | **0.331** |
| mAP@0.7 | 0.100 | 0.089 | 0.088 | 0.099 | 0.095 | 0.099 | **0.102** |
| AP car | 0.626 | 0.611 | 0.589 | 0.633 | 0.622 | 0.620 | **0.650** |
| AP truck | 0.557 | 0.497 | 0.462 | 0.521 | **0.568** | 0.513 | 0.541 |
| AP bus | 0.357 | 0.477 | **0.673** | 0.602 | 0.347 | 0.584 | 0.637 |
| AP pedestrian | 0.334 | **0.418** | 0.218 | 0.346 | 0.408 | 0.289 | 0.352 |
| AP bicycle | 0.204 | 0.120 | 0.175 | 0.169 | 0.202 | 0.203 | **0.205** |
| AP motorcycle | **0.300** | 0.000 | 0.100 | 0.100 | 0.167 | 0.133 | 0.100 |
| AP traffic_cone | **0.490** | — | 0.151 | 0.151 | **0.490** | 0.335 | 0.335 |
| AP barrier | 0.007 | — | 0.159 | 0.159 | 0.007 | **0.161** | **0.161** |
| AP construction_vehicle | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Flag recall | 0.786 | 0.604 | 0.704 | 0.772 | 0.799 | 0.790 | **0.820** |
| Flag precision | 0.712 | **0.918** | 0.850 | 0.894 | 0.862 | 0.812 | 0.871 |
| Tỉ lệ nhóm low | 0.412 | 0.552 | 0.437 | 0.366 | 0.380 | 0.362 | 0.327 |
| Lỗi lọt trong nhóm low | **0.277** | 0.490 | 0.459 | 0.457 | 0.353 | 0.379 | 0.393 |
| Object cần sửa | **706** | 1055 | 1086 | 1611 | 1301 | 1144 | 1614 |

AP 0.000 của YOLO26-L ở cone / barrier / construction_vehicle ghi "—": COCO không có các lớp đó.

## Đọc bảng

- YOLO26-L và YOLOE-26-L bổ sung cho nhau đúng như thiết kế: YOLO26-L mạnh ở pedestrian, YOLOE-26-L mạnh ở bus và
  là model duy nhất bắt được barrier (0.16 so với 0.007 của YOLO-World).
- Mặc định là YOLOE-26-L một mình (một model phủ đủ 10 lớp): mAP@0.5 0.281 so với 0.320 của YOLO-World. Mạnh hơn rõ
  ở bus (0.673 so với 0.357) và barrier (0.159 so với 0.007); yếu hơn rõ ở **traffic_cone** (0.151 so với 0.490),
  motorcycle và pedestrian.
- Thêm YOLO-World vào làm model thứ ba cho mAP@0.5 cao nhất (0.331) và flag recall cao nhất (0.820), đổi lại chạy 3
  model / ảnh.
- Mọi lựa chọn mới, kể cả YOLOE-26-L một mình, có **tỉ lệ lỗi lọt trong nhóm low cao hơn** YOLO-World (0.35–0.49 so
  với 0.28; YOLOE-26-L: 0.459) và sinh nhiều box sai hơn: ngưỡng risk hiện chỉnh cho điểm số của YOLO-World. Trước khi
  bật "Approve all low-risk" cho model mới nên chỉnh lại ngưỡng (`qa.risk.levels`, `detection.score_threshold`) và
  dùng audit ngẫu nhiên ở tab QC để đo tỉ lệ lỗi thật.
- GT là hộp bao chiếu từ 3D (rộng hơn box sát vật), nên mAP@0.7 thấp là bình thường; nuScenes mini chỉ 79 keyframe
  nên chênh lệch nhỏ (≈ 0.01) chưa đủ để kết luận.
