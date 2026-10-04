DAM4SAM: model `sam21pp-L`, stride 3, tăng tốc bật; scene: scene-0003, scene-0012, scene-0013, scene-0014, scene-0015, scene-0016, scene-0017, scene-0018, scene-0036, scene-0038, scene-0039, scene-0092, scene-0093, scene-0094, scene-0095, scene-0096, scene-0098, scene-0099, scene-0100, scene-0102.

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| flow+byte | 3438 | 1161 | 171 | 853 | 0.563 | 0.457 | 0.750 | 0.453 | 0.705 | 731.2 |
| flow+botsort | 3460 | 1146 | 161 | 853 | 0.564 | 0.459 | 0.751 | 0.459 | 0.709 | 1089.7 |
| dam4sam+botsort | 3678 | 1315 | 155 | 639 | 0.573 | 0.472 | 0.761 | 0.467 | 0.725 | 2456.2 |

**Khuyến nghị: `flow+botsort`** (HOTA 0.564, IDF1 0.709, 1089.7 s).
`dam4sam+botsort` có HOTA cao nhất (0.573) nhưng chỉ hơn 0.009 (< 0.01), không đáng thêm độ phức tạp / thời gian (2456.2 s).
So với mặc định `flow+byte`: HOTA 0.563 → 0.564, IDF1 0.705 → 0.709, nhãn đúng 3438 → 3460, thời gian ×1.
Quy tắc chọn: HOTA cao nhất; chênh dưới 0.01 thì lấy cấu hình đơn giản / nhanh hơn. Chỉ chọn trên dev.
