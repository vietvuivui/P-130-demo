# Tracking 2D — so sánh các luồng lan truyền (TrackEval)

Cách đo: nhãn gốc nuScenes ở một keyframe đóng vai nhãn người đã duyệt, lan truyền tối đa 10 keyframe, chấm với nhãn
gốc ở các keyframe sau. Hai tập: dev (3 scene) và held-out (20 scene); CAM_FRONT; detector YOLOE-26-L fine-tune; RTX 4050;
chạy 2026-10-02. DAM4SAM: model `sam21pp-L`, stride 3 (keyframe + 1 ảnh giữa
hai keyframe), dùng chung image encoder + autocast.

## Kết luận

- **Mặc định giữ optical flow + ByteTrack** (luồng "Nhanh"): chạy CPU, không cần cài thêm gì.
- **DAM4SAM + BoT-SORT (luồng "Chính xác") hơn một chút trên held-out**: HOTA 0.563 → 0.573, IDF1 0.705 → 0.725, nhãn đúng
  +7%, mất dấu −25%; đổi lại box sai +13%, cần GPU và chậm hơn 3,4 lần. Tỉ lệ box đúng trên box ghi ra gần như không đổi
  (72,1% → 71,4%): DAM4SAM giữ được nhiều vật hơn chứ không chính xác hơn trên từng box.
- Trên 3 scene dev hai luồng ngang nhau (0.580 và 0.578), nên mức hơn này nhỏ; chưa có khoảng tin cậy theo từng scene.

## Held-out: 20 scene, 795 keyframe (workspace `data/eval_temporal/ws_heldout`)

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| optical flow + ByteTrack (mặc định) | 3438 | 1161 | 171 | 853 | 0.563 | 0.457 | 0.750 | 0.453 | 0.705 | 731 |
| optical flow + BoT-SORT | 3460 | 1146 | 161 | 853 | 0.564 | 0.459 | 0.751 | 0.459 | 0.709 | 1090 |
| **DAM4SAM + BoT-SORT** | 3678 | 1315 | 155 | 639 | **0.573** | 0.472 | 0.761 | 0.467 | **0.725** | 2456 |

BoT-SORT đổi 83 / 9981 lần ghép khi đi với optical flow, 79 / 3772 khi đi với DAM4SAM. Chưa đo trên held-out: DAM4SAM +
ByteTrack, DAM4SAM thuần, cấu hình lai.

## Dev: 3 scene (scene-0035, scene-0097, scene-0101)

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| **optical flow + ByteTrack (mặc định)** | 846 | 209 | 32 | 201 | **0.580** | 0.481 | 0.733 | 0.546 | **0.757** | 62 |
| optical flow + BoT-SORT | 846 | 209 | 28 | 203 | 0.580 | 0.479 | 0.736 | 0.548 | 0.758 | 98 |
| lai: flow, DAM4SAM cho vật mất detection + ByteTrack | 859 | 255 | 67 | 173 | 0.562 | 0.476 | 0.695 | 0.498 | 0.730 | 506 |
| DAM4SAM + ByteTrack | 849 | 261 | 20 | 200 | 0.574 | 0.470 | 0.728 | 0.505 | 0.740 | 848 |
| DAM4SAM + BoT-SORT | 856 | 248 | 18 | 196 | 0.578 | 0.475 | 0.733 | 0.516 | 0.747 | 731 |
| DAM4SAM thuần (không ghép detection) ¹ | 1007 | 691 | 83 | 0 | 0.532 | 0.428 | 0.700 | 0.234 | 0.685 | 1029 |

¹ Đo trước khi sửa ngưỡng dừng track (xem dưới); cấu hình này không dùng ngưỡng theo detection nên vẫn so được.

## Nhận xét (dev)

- **Nút thắt là detector.** DetA của mọi cấu hình có ghép detection gần bằng nhau (0.470–0.481): box ghi ra ở keyframe là
  box của YOLO, tracker chỉ quyết định box nào thuộc vật nào, và optical flow đã làm việc đó đủ tốt.
- **BoT-SORT gần như không đổi gì khi đã có optical flow** (đổi 8 / 1770 lần ghép): flow đã bù chuyển động camera. Trong
  cấu hình DAM4SAM, phần còn tác dụng là so ngoại hình (histogram màu), đổi 19 / 666 lần ghép.
- **DAM4SAM thuần tệ nhất**: không có detection thì không biết dừng khi vật khuất, và hai tracker có thể bám cùng một vật.
- **DAM4SAM + BoT-SORT**: ít đổi ID hơn mặc định (18 so với 32) nhưng nhiều box sai hơn (248 so với 209); HOTA ngang.
- **Cấu hình lai**: đổi ID gấp đôi (67). Giả thuyết chưa kiểm chứng: SAM được gọi sau một quãng không theo dõi, box nhảy
  sang hộp bao mask rồi khớp nhầm detection của vật khác.
- Chênh lệch giữa các cấu hình DAM4SAM (0.562–0.578) nhỏ và chỉ đo trên 3 scene dev, nên thứ hạng giữa chúng không chắc.

## Ngưỡng dừng track khi dùng stride

Track dừng sau `max_coast_images` = 6 ảnh liên tiếp không khớp detection. Với stride 3, 6 ảnh là khoảng 1,5 giây thay vì
0,5 giây, nên track sống quá lâu sau khi vật khuất. Đã sửa: ngưỡng chia theo stride.

| Cấu hình | Box sai | Mất dấu | HOTA |
|---|---|---|---|
| DAM4SAM + ByteTrack, trước → sau khi sửa | 418 → 261 | 70 → 200 | 0.562 → 0.574 |
| DAM4SAM + BoT-SORT, trước → sau khi sửa | 371 → 248 | 70 → 196 | 0.571 → 0.578 |

## Chưa đo

- Khoảng tin cậy / kết quả theo từng scene trên held-out; các cấu hình DAM4SAM khác trên held-out.
- DAM4SAM ở mọi ảnh 12 Hz (stride 1) trên cả 3 scene, và model nhỏ hơn (`sam21pp-T`).
- Detector sau khi fine-tune thêm trên RTX 3090.

Chạy lại held-out: `python tools2d\dam4sam.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_heldout --out
eval\results\trackall_heldout --stride 3 --configs flow+byte flow+botsort dam4sam+botsort`.

Chạy lại dev: `python tools2d\dam4sam.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_dev --out
eval\results\trackall_v2 --stride 3 --scenes scene-0035 scene-0097 scene-0101`.
