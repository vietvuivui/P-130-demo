DAM4SAM: model `sam21pp-L`, stride 3, tăng tốc bật; scene: scene-0035, scene-0097, scene-0101.

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| flow+byte | 846 | 209 | 32 | 201 | 0.580 | 0.481 | 0.733 | 0.546 | 0.757 | 81.3 |
| flow+botsort | 846 | 209 | 28 | 203 | 0.580 | 0.479 | 0.736 | 0.548 | 0.758 | 97.8 |
| dam4sam+none | 1007 | 691 | 83 | 0 | 0.532 | 0.428 | 0.700 | 0.234 | 0.685 | 1029.3 |
| dam4sam+byte | 923 | 418 | 39 | 70 | 0.562 | 0.467 | 0.706 | 0.425 | 0.724 | 979.4 |
| dam4sam+botsort | 924 | 371 | 37 | 70 | 0.571 | 0.483 | 0.705 | 0.466 | 0.739 | 883.1 |

**Khuyến nghị: `flow+byte`** (HOTA 0.580, IDF1 0.757, 81.3 s).
`flow+botsort` có HOTA cao nhất (0.580) nhưng chỉ hơn 0.000 (< 0.01), không đáng thêm độ phức tạp / thời gian (97.8 s).
Ghép DAM4SAM với detection (byte) so với DAM4SAM thuần: HOTA 0.532 → 0.562, box sai 691 → 418, mất dấu 0 → 70.
Thêm BoT-SORT vào DAM4SAM: HOTA 0.562 → 0.571, đổi ID 39 → 37 (ngoại hình đổi 29/720 lần ghép).
Quy tắc chọn: HOTA cao nhất; chênh dưới 0.01 thì lấy cấu hình đơn giản / nhanh hơn. Chỉ chọn trên dev.
