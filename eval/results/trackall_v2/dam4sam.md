DAM4SAM: model `sam21pp-L`, stride 3, tăng tốc bật; scene: scene-0035, scene-0097, scene-0101.

| Cấu hình | Nhãn đúng | Box sai | Đổi ID | Mất dấu | HOTA | DetA | AssA | MOTA | IDF1 | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| flow+byte | 846 | 209 | 32 | 201 | 0.580 | 0.481 | 0.733 | 0.546 | 0.757 | 62.2 |
| flow+dam4sam+byte | 859 | 255 | 67 | 173 | 0.562 | 0.476 | 0.695 | 0.498 | 0.730 | 506.0 |
| dam4sam+byte | 849 | 261 | 20 | 200 | 0.574 | 0.470 | 0.728 | 0.505 | 0.740 | 847.5 |
| dam4sam+botsort | 856 | 248 | 18 | 196 | 0.578 | 0.475 | 0.733 | 0.516 | 0.747 | 730.9 |

**Khuyến nghị: `flow+byte`** (HOTA 0.580, IDF1 0.757, 62.2 s).
Thêm BoT-SORT vào DAM4SAM: HOTA 0.574 → 0.578, đổi ID 20 → 18 (ngoại hình đổi 19/666 lần ghép).
Quy tắc chọn: HOTA cao nhất; chênh dưới 0.01 thì lấy cấu hình đơn giản / nhanh hơn. Chỉ chọn trên dev.
