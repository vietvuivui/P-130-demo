# Kịch bản demo — AutoLabel 2D (chế độ Ảnh + chế độ Video)

Hai cách chạy, cùng một UI:

| | Demo nhanh | Demo nuScenes thật |
|---|---|---|
| Cần gì | `pip install -r requirements.txt` | GPU + nuScenes v1.0-mini, hoặc zip tải từ notebook Kaggle |
| Lệnh | `python -m src.demo` (hoặc `make demo`) | `python -m src.cli run --scenes scene-0061 scene-0103` rồi `uvicorn src.main:app` |
| Dữ liệu | video tổng hợp 16 s, detector theo màu | CAM_FRONT + LiDAR thật, YOLO-World |
| Workspace | `data/demo/workspace` (riêng) | `data/workspace` |

Trình diễn tổng cộng khoảng 5 phút (giới hạn video demo của BTC). Thời lượng từng đoạn ghi trong ngoặc.

---

## 0. Chuẩn bị (trước khi lên sân khấu)

```bash
python -m src.demo --reset      # sinh lại dữ liệu sạch (~30 s), mở http://localhost:8000
```

Mở sẵn trình duyệt ở `http://localhost:8000`, ô *Người duyệt* điền tên. Nếu muốn demo nút tải lên thì để sẵn file
`data/demo/street_demo.mp4` ở chỗ dễ chọn.

## 1. Vấn đề (30 s)

Gán nhãn 2D cho xe tự lái: người phải soát từng box ở từng frame. Một scene 20 giây ở 2 Hz đã là 40 frame, mỗi
frame hàng chục object. Hệ thống làm hai việc để giảm việc của người:

1. **Review by exception** — model tự sinh box, QA Agent chấm rủi ro, người chỉ xem kỹ box rủi ro cao.
2. **Lan truyền nhãn** — người duyệt xong một frame, nhãn được mang sang các frame sau của video.

## 2. Chế độ 🖼 Ảnh — MVP review by exception (1 phút)

1. Thanh trên cùng đang ở **🖼 Ảnh**. Hàng đợi bên trái sắp theo risk: frame khó nhất lên đầu.
2. Mở frame đầu hàng đợi. Chỉ vào ba nhóm bên phải: High (xem kỹ), Medium, Low (thu gọn, duyệt theo lô).
3. Bấm vào một card medium để xem issue code và lời giải thích, ví dụ:
   - `CLASS_CONFLICT` — xe van cam, detector phân vân truck/car;
   - `LOW_CONFIDENCE` — xe xanh nhỏ ở xa;
   - `FLICKER` — vệt đỏ loé lên, không có ở sweep lân cận (dải *Temporal consistency* bên dưới cho thấy
     `missing` ở t±1, t±2).
4. Nói: đây là MVP của Huy — duyệt từng ảnh độc lập, không có lan truyền.

## 3. Chế độ 🎞 Video — timeline + lan truyền (2 phút 30)

1. Bấm **🎞 Video**. Bên trái là danh sách video (scene nuScenes hoặc mp4 tải lên), dưới khung ảnh là **timeline**
   của video: mỗi keyframe một thẻ, có thời điểm, số object chờ duyệt, vạch màu risk.
2. Frame #1 (0.0 s) đang mở. Sửa như chế độ Ảnh:
   - xe van cam: chọn `car` → **Đổi lớp**;
   - xe xanh nhỏ ở xa: **Keep** (đúng là xe, chỉ là xa);
   - mở nhóm *Low risk*, chọn poster hình người trên tường (`pedestrian`) → **Delete** (QA không bắt được lỗi
     này vì video không có LiDAR; người bắt được);
   - **Approve all low-risk** (`A`) → **Approve frame** (`Enter`).
3. Ô *Lan truyền khi approve* đang bật nên ngay khi approve:
   - toast "Lan truyền sang 20 frame …"; timeline hiện ★ ở frame #1 và ↦ ở 20 frame sau (giới hạn
     `propagation.max_keyframes`; approve frame nào thì lan truyền tiếp từ frame đó);
   - UI tự mở frame #2. Xe van giờ là `car ↦ c 0.9x`: lớp người sửa đã được mang theo. Poster nằm ở *Đã xử lý*
     với nhãn "✗ Tự xoá theo frame #1" (bấm Keep nếu muốn khôi phục).
   - Hầu hết nhãn lan truyền rơi vào Low → một phím `A` là xong frame.
4. **Kéo** thẻ frame #13 (6.0 s) từ timeline **thả lên khung ảnh**. Xe đỏ đang đi sau biển quảng cáo, chỉ lộ một
   dải hẹp → box lan truyền bị gắn `ASPECT_RATIO_ABNORMAL`, nằm ở Medium để người xem. Sửa box bằng **✎ Sửa box**
   hoặc Delete tuỳ quy ước.
5. Kéo tiếp frame #14–#15: xe đỏ bị che hẳn nên track dừng; khi xe hiện lại ở bên kia biển thì là object mới cần
   duyệt (giới hạn đã biết, xem README).
6. Từ frame #14 có xe đạp vào khung hình: object mới, không có ở frame #1, nên không có tag ↦ — người duyệt như
   object detector sinh bình thường.
7. Bấm **▶ Phát** (`Space`) để xem nhanh cả video với nhãn hiện tại, bấm lần nữa để dừng.

## 4. Tải lên mp4 (30 s)

1. **⬆ Tải lên mp4** → chọn `data/demo/street_demo.mp4` (hoặc video dashcam bất kỳ).
2. Server cắt frame ngay (10 fps, keyframe 2 fps); danh sách bên trái hiện "Đang xử lý x%", video đang xem không bị
   đổi. Xong thì toast báo, bấm vào video mới để duyệt.
3. Nói: cùng pipeline với nuScenes (detect → QA Agent → review → lan truyền); không có LiDAR thì check LiDAR tự bỏ
   qua.

## 5. Kết quả (30 s)

1. Tab **Metrics & Export**: M1 (thời gian/frame), M4 (tỉ lệ nhãn phải sửa), ô *Nhãn lan truyền phải sửa*
   (bao nhiêu nhãn lan truyền người phải sửa, bao nhiêu box tự xoá), flag precision/recall của QA Agent.
2. **Xuất dataset**: chỉ frame đã approve, COCO + JSONL, mỗi object có `track_id` và `propagated_from`.
3. Tab **Correction log**: mỗi thao tác một dòng, dùng để đo và chỉnh ngưỡng.

---

## Bản nuScenes thật

```bash
python -m src.cli run --scenes scene-0061 scene-0103   # GPU; hoặc giải nén zip từ notebook Kaggle
python -m src.cli detect-sweeps --scenes scene-0061    # tuỳ chọn: detection ở mọi ảnh 12Hz cho tracker
uvicorn src.main:app --port 8000
```

Kịch bản giống hệt, khác ở chỗ:

- Chế độ Video liệt kê mỗi scene là một video; tracker đi qua mọi ảnh CAM_FRONT 12 Hz giữa hai keyframe.
- Bật **LiDAR** (`L`) để thấy điểm LiDAR chiếu lên ảnh; issue `NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH` xuất hiện.
- Bật **GT** (`G`) để so với nhãn nuScenes.
- `python -m src.cli eval-propagation` → `eval/results/propagation_eval.md`: lan truyền so với GT khi keyframe
  đầu hoàn hảo (không cần người gán).

## Khi có sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Chế độ Video trống | Chưa có dữ liệu: `python -m src.demo --reset` hoặc `python -m src.cli run --scenes …` |
| Lan truyền báo `DATA_UNAVAILABLE` | Scene nuScenes cần bảng nuScenes trên máy; kiểm tra `NUSCENES_DATAROOT` |
| Tải lên báo `VIDEO_UNREADABLE` | Đổi sang mp4 H.264 (`ffmpeg -i in.mov -c:v libx264 out.mp4`) |
| Muốn làm lại từ đầu | `python -m src.demo --reset` (chỉ xoá `data/demo/`) |
