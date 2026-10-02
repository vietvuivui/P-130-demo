# Tracking 2D — so sánh các luồng lan truyền (TrackEval)

Cách đo: nhãn gốc nuScenes ở một keyframe đóng vai nhãn người đã duyệt, lan truyền tối đa 10 keyframe, chấm với nhãn
gốc ở các keyframe sau. Workspace `data/eval_temporal/ws_dev` (CAM_FRONT, detector YOLOE-26-L fine-tune), RTX 4050.
Chạy lại: `scripts\tasks.ps1 trackall` (một scene: thêm `-Scenes scene-0035`).

## 3 scene dev (scene-0035, scene-0097, scene-0101), chạy 2026-10-02

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| optical flow + ByteTrack (luồng "Nhanh", mặc định) | 846 | 209 | 32 | 201 | 0.580 | 0.481 | 0.733 | 0.546 | 0.757 | 119 |
| optical flow + BoT-SORT | 846 | 209 | 28 | 203 | 0.580 | 0.479 | 0.736 | 0.548 | 0.758 | 136 |

BoT-SORT đổi kết quả ở 8 / 1770 lần ghép: optical flow đã bù chuyển động camera nên phần bù của BoT-SORT gần như thừa.

## scene-0035 (DAM4SAM bản chưa tăng tốc, chưa xuất HOTA)

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | Giây |
|---|---|---|---|---|---|
| optical flow + ByteTrack | 106 | 31 | 10 | 17 | 14 |
| optical flow + BoT-SORT | 106 | 31 | 10 | 17 | 18 |
| DAM4SAM + ByteTrack | 117 | 31 | 10 | 9 | 1466 |
| DAM4SAM + BoT-SORT | 120 | 26 | 5 | 9 | 1226 |

Trên scene này optical flow + ByteTrack đạt HOTA 0.552 / IDF1 0.739.

## Chưa đo

- HOTA / IDF1 của các cấu hình DAM4SAM.
- DAM4SAM không kèm tracker ghép (`dam4sam+none`): DAM4SAM tự bám từng vật, nên bước ghép có thể thừa.
- Bản DAM4SAM tăng tốc (dùng chung image encoder + autocast) trên cả 3 scene. Phép thử nhỏ trên một vật: 33.5 → 9.5 giây.
- Tập held-out. Các số trên là tập dev.
