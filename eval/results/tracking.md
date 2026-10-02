# Tracking 2D — so sánh các luồng lan truyền (TrackEval)

Cách đo: nhãn gốc nuScenes ở một keyframe đóng vai nhãn người đã duyệt, lan truyền tối đa 10 keyframe, chấm với nhãn
gốc ở các keyframe sau. Workspace `data/eval_temporal/ws_dev` (3 scene dev: scene-0035, scene-0097, scene-0101; CAM_FRONT;
detector YOLOE-26-L fine-tune), RTX 4050, chạy 2026-10-02. DAM4SAM: model `sam21pp-L`, stride 3 (keyframe + 1 ảnh giữa
hai keyframe), dùng chung image encoder + autocast.

**Kết luận: giữ optical flow + ByteTrack làm mặc định.** Không cấu hình nào vượt được nó; các cấu hình DAM4SAM chậm hơn
8–14 lần.

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| **optical flow + ByteTrack (mặc định)** | 846 | 209 | 32 | 201 | **0.580** | 0.481 | 0.733 | 0.546 | **0.757** | 62 |
| optical flow + BoT-SORT | 846 | 209 | 28 | 203 | 0.580 | 0.479 | 0.736 | 0.548 | 0.758 | 98 |
| lai: flow, DAM4SAM cho vật mất detection + ByteTrack | 859 | 255 | 67 | 173 | 0.562 | 0.476 | 0.695 | 0.498 | 0.730 | 506 |
| DAM4SAM + ByteTrack | 849 | 261 | 20 | 200 | 0.574 | 0.470 | 0.728 | 0.505 | 0.740 | 848 |
| DAM4SAM + BoT-SORT | 856 | 248 | 18 | 196 | 0.578 | 0.475 | 0.733 | 0.516 | 0.747 | 731 |
| DAM4SAM thuần (không ghép detection) ¹ | 1007 | 691 | 83 | 0 | 0.532 | 0.428 | 0.700 | 0.234 | 0.685 | 1029 |

¹ Đo trước khi sửa ngưỡng dừng track (xem dưới); cấu hình này không dùng ngưỡng theo detection nên vẫn so được.

## Nhận xét

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

- Tập held-out (các số trên là tập dev).
- DAM4SAM ở mọi ảnh 12 Hz (stride 1) trên cả 3 scene, và model nhỏ hơn (`sam21pp-T`).
- Detector sau khi fine-tune thêm trên RTX 3090.

Chạy lại: `python tools2d\dam4sam.py --dataroot ..\v1.0-trainval --workspace data\eval_temporal\ws_dev --out
eval\results\trackall_v2 --stride 3 --scenes scene-0035 scene-0097 scene-0101`.
