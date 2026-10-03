# Kịch bản video MVP 3 phút (Gate G2) — user flow end-to-end

Chuẩn bị: `pip install -r requirements.txt`, `python scripts/download_sam_onnx.py`, chạy
`$env:AUTH_REQUIRED="1"; python -m uvicorn src.main:app --port 8000`, mở `http://localhost:8000`. Chuẩn bị sẵn một dự án
nuScenes đã xử lý xong (`nusc3d_scene-0035.zip`) để không phải chờ khi quay, và một clip mp4 ngắn để tải lên.
Quay màn hình 1920×1080, thu tiếng thuyết minh.

| Thời gian | Trên màn hình | Lời nói (ý chính) |
|---|---|---|
| 0:00–0:15 | Trang đăng nhập → trang Dự án | Vấn đề: gán nhãn 2D + 3D cho xe tự lái tốn công; AutoLabel 3D để máy gán trước, người chỉ duyệt chỗ rủi ro |
| 0:15–0:40 | Tạo dự án, tải lên clip mp4 / zip nuScenes, thanh tiến độ auto-label | Tải dữ liệu lên, hệ thống cắt frame, detect, QA Agent chấm rủi ro |
| 0:40–0:55 | Thẻ dự án → Thành viên → tạo link mời → Chia việc | Mời người khác bằng link, chia frame; frame đang mở bị khoá 🔒 |
| 0:55–1:35 | Chế độ Ảnh: hàng đợi khó nhất trước, box đỏ + lý do, Keep / Delete / đổi lớp, *Approve all low-risk* | Review by exception: chỉ xem box rủi ro, box an toàn duyệt theo lô |
| 1:35–1:55 | `M` Chọn vật: bấm vào một xe chưa có nhãn → mask + box → Lưu | Thêm nhãn bằng một cú bấm (SAM 2.1), không cần vẽ box |
| 1:55–2:20 | Chế độ Video: chọn luồng Nhanh / Chính xác, Approve → nhãn ↦ sang keyframe sau, phát video | Lan truyền nhãn đã duyệt sang các frame sau; người dùng tự chọn luồng |
| 2:20–2:40 | Chế độ 3D: box 3D, bấm box trên ảnh camera → nhảy tới box trong 3D, box bị che vẽ nét đứt | Nhãn 3D từ ensemble LiDAR, kiểm chứng bằng 6 camera |
| 2:40–3:00 | Tab Metrics (tỉ lệ phải sửa, HOTA / IDF1) → Xuất COCO / nuScenes | Số đo và dataset xuất ra chỉ gồm frame đã duyệt |

Số để nói khi chốt: detector 2D mAP50 0.366; 3D mAP 0.668 / NDS 0.713; lan truyền 2D 77% box đúng vật, HOTA 0.563 (Nhanh) / 0.573 (DAM4SAM) trên 20 scene held-out.
