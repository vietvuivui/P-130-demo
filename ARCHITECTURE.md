# Architecture — AutoLabel 2D

## System Overview

Auto-label 2D box trên ảnh CAM_FRONT của nuScenes bằng model open-vocab (không train), rồi một
QA Agent kiểm chứng chéo mỗi box bằng ba nguồn độc lập — confidence của model, point cloud LiDAR,
và các sweep camera 12Hz lân cận — để chấm risk. Người vẫn duyệt 100% nhãn, nhưng nhãn risk thấp
duyệt theo lô, thời gian dồn vào số ít nhãn risk cao (review by exception). Mọi thao tác được log
và chỉ frame đã approve mới được xuất.

Sơ đồ: [docs/architecture_diagram.md](docs/architecture_diagram.md).

## Data Flow

1. `python -m src.cli run` lấy keyframe (+ sweep t-2..t+2, LiDAR, calibration) từ `v1.0-mini`.
2. Detector chạy trên keyframe và sweep; kết quả thô cache ở `data/workspace/cache/detections/`.
3. QA Agent (LangGraph) chạy 3 check song song → sinh issue → tính risk.
4. Frame JSON ghi vào `data/workspace/frames/`, kèm LiDAR đã chiếu (`lidar/`) và GT 2D (`gt/`).
5. `uvicorn src.main:app` phục vụ API + UI. Người duyệt; mỗi thao tác ghi `corrections.jsonl`.
6. `POST /api/v1/export` ghi dataset vào `data/workspace/exports/<id>/`.

Video mp4 tải lên (`POST /api/v1/videos/upload`, `src/services/video.py`) đi cùng đường, chỉ khác bước 1:

1. OpenCV cắt frame ở `video.track_fps` (10 fps) vào `data/workspace/videos/<video_id>/frames/`; cứ
   `track_fps / label_fps` frame (mặc định 5) là một keyframe. Record `videos/<video_id>.json` giữ timeline, đóng
   vai `sample_data` next-chain của nuScenes. Đường dẫn ảnh ghi dạng `@workspace/videos/...`.
2. Bước 2–4 chạy nền (FastAPI `BackgroundTasks`) bằng `pipeline.label_keyframe`, hàm dùng chung với nuScenes:
   sweep = các frame t±1, t±2 quanh keyframe; không có LiDAR nên check LiDAR tự bỏ qua; intrinsic danh nghĩa
   (tiêu cự = chiều rộng ảnh).
3. Frame ID `<video_id>_NNN`, `scene = video_id`: UI và lan truyền coi mỗi video như một scene.

## UI: hai chế độ

| | 🖼 Ảnh | 🎞 Video |
|---|---|---|
| Danh sách trái | Hàng đợi frame theo risk (MVP) | Video: scene nuScenes (suy từ frame) + mp4 tải lên (`GET /videos`) |
| Dưới khung ảnh | Dải temporal t-2 … t+2 | Timeline của video (`GET /videos/{id}`): kéo thẻ thả lên khung ảnh để mở |
| Approve xong | Mở frame rủi ro cao nhất tiếp theo | Lan truyền (nếu bật) rồi mở frame kế tiếp chưa duyệt của video |
| Lan truyền | Không | Tự chạy khi approve, hoặc nút ↦ / phím `T` |

Khung chỉnh sửa (canvas + panel risk) là một, dùng chung cho cả hai chế độ.

## QA Agent

| Check | Issue code | Cách tính |
|---|---|---|
| 3.1 Confidence | `LOW_CONFIDENCE` | score < 0.35 |
| | `CLASS_CONFLICT` | cùng vị trí (IoU ≥ 0.55) có lớp khác với score ≥ 0.6 × lớp chính |
| 3.2 LiDAR | `NO_LIDAR_SUPPORT` | < 3 điểm LiDAR trong box cao ≥ 60px |
| | `SIZE_DEPTH_MISMATCH` | cao ước lượng `h_px · depth / f_y` ngoài khoảng hợp lý của lớp (× dung sai) |
| 3.3 Temporal | `FLICKER` | box keyframe xuất hiện lại ở < 2 sweep lân cận |
| | `RECOVERED_BY_TRACK` | object có ở sweep trước + sau, sót ở keyframe → đề xuất box nội suy |
| Hình học | `BOX_TOO_LARGE` | box > 35% diện tích ảnh |
| | `ASPECT_RATIO_ABNORMAL` | tỉ lệ rộng/cao ngoài khoảng của lớp (bỏ qua box chạm mép) |

`risk = w1(1 − score) + w2·lidar + w3·temporal + w4·geometric`, mỗi thành phần ∈ [0, 1],
trọng số chuẩn hoá về tổng 1. Object có bất kỳ issue nào bị nâng tối thiểu lên 0.30 để không
lọt vào nhóm duyệt theo lô. Nhóm: low < 0.30 ≤ medium < 0.60 ≤ high. Mọi ngưỡng ở
`configs/autolabel.yaml`.

## Lan truyền nhãn trên video

Người approve một keyframe → `POST /frames/{id}/propagate` (UI tự gọi khi bật "Lan truyền khi approve").

1. **Khởi tạo track** từ quyết định của người: object đã duyệt → track "keep" (lớp khoá theo người);
   object máy sinh bị xoá → track "suppress". Object còn pending không lan truyền. Mỗi object nhận
   `track_id` = `<keyframe>:<object_id>`, giữ nguyên qua các frame.
2. **Tracker** đi qua mọi ảnh sau keyframe — CAM_FRONT 12Hz (`NuScenesMini.camera_timeline`) hoặc mọi frame
   10 fps của video tải lên (timeline trong record; `sequence.WorkspaceSequenceSource` chọn nguồn), dự đoán box
   theo vận tốc không đổi trong toạ độ ảnh, ghép với detection đã cache (IoU ≥ 0.3, một-một). Ảnh chưa có
   trong cache: chỉ dự đoán. Dừng track khi > 6 ảnh liền không khớp, ra khỏi khung, hoặc box quá nhỏ.
3. **Ở mỗi keyframe đích còn "auto"**: tính c_prop; track nhận box pre-label trùng nó (IoU ≥ 0.5) → object
   `source="propagated"`, lớp của người, box của detector ở chính frame đó. Detector không thấy → thêm box
   dự đoán (`PROP_COASTING`). Track "suppress" chỉ tự xoá box khớp chặt (IoU ≥ 0.5) và cùng lớp.
4. **Dừng** trước frame đầu tiên có status khác "auto" (người đã mở/duyệt), hết scene, hoặc đủ `max_keyframes`.

`c_prop = (0.5·agreement + 0.3·continuity + 0.2·lidar) / Σw × 0.99^(hops−1)`, trong đó agreement = IoU
box dự đoán–detection ở frame đó (0 nếu detector không thấy), continuity = tỉ lệ ảnh có khớp kể từ keyframe,
lidar = số điểm LiDAR trong box so với keyframe (chặn 1). Nhãn lan truyền đi qua cùng risk scoring:
thành phần detection = 1 − c_prop; `PROP_LOW_CONF` khi c_prop < 0.6.

| Quy tắc | Vì sao |
|---|---|
| Chỉ lan truyền từ frame đã approve | Chỉ mang đi quyết định đã chốt của người |
| Không ghi vào frame status khác "auto" | Không bao giờ ghi đè công người đã làm |
| Luôn dựng lại từ `frame.prelabel` | Lan truyền lại (từ keyframe khác) không bị cộng dồn |
| Lớp detector khác lớp người chốt không phải là lỗi | Người đã sửa lớp ở keyframe; chỉ cờ `PROP_CLASS_DIFFERS` khi lớp detector đổi giữa chừng (dấu hiệu track nhảy object) |
| M4 tách theo nguồn, bỏ object tự xoá | Tự xoá là máy làm, không phải người sửa |

Đánh giá không cần người: `python -m src.cli eval-propagation` lấy GT keyframe đầu mỗi scene làm "nhãn người",
lan truyền, so với GT cùng `instance_token` ở các keyframe sau (tỉ lệ đúng, đổi ID, độ phủ, c_prop có tách được
đúng/sai không).

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Loader dữ liệu | Tự viết, chỉ numpy | nuscenes-devkit kéo nhiều dependency, pipeline chỉ cần vài bảng |
| Detector mặc định | YOLOE-26-L (trước đây YOLO-World-L) | Open-vocab, nhận được barrier/cone; trên 2 scene trainval ngẫu nhiên mAP@0.5 0.45 so với 0.31 (YOLO-World từ vựng COCO) và 0.40 (YOLO26), lan truyền 50/52 đúng — `eval/compare_detectors.ipynb` |
| Grounding DINO | bản open-weight tiny | Bản 1.5 Edge chỉ có qua API |
| QA Agent | LangGraph, deterministic | Cần chạy trong batch và test; không có câu hỏi mở nào cần LLM |
| Lưu trữ | File JSON + JSONL | 404 frame, 1 người duyệt; không cần DB cho demo, dễ diff/export |
| UI | HTML/JS thuần do FastAPI phục vụ | Không cần build step, 1 lệnh là chạy |
| GT 2D | Hộp bao 8 đỉnh box 3D chiếu xuống | nuScenes không có box 2D gốc; báo cáo AP@0.5 là chính |
| Lan truyền 2D | Track qua sweep 12Hz bằng detection đã cache | Keyframe 2Hz quá thưa để khớp trực tiếp; không cần model mới (SAM2 video tốn GPU); `track_id` để gắn 3D sau |
| Box lan truyền không có detection đỡ | Không ghi ra (`emit_coasting: false`), track vẫn chạy | Chỉ ~17% box thuần dự đoán đúng; ghi ra làm người phải xoá nhiều hơn công lan truyền tiết kiệm — `eval/review_simulation.ipynb` |
| Ngưỡng giữ box | `min_score: 0.30` sau fusion | Precision 0.34 → 0.51, box bị gắn cờ 260 → 88 trên 40 keyframe, không cần train — `eval/precision_tuning.ipynb` |
| Video tải lên | Cắt thành frame + keyframe giống nuScenes | Dùng lại nguyên pipeline, QA Agent, review và lan truyền; không cần code riêng cho video |
| Xử lý video tải lên | `BackgroundTasks` trong process FastAPI | Đủ cho 1 người duyệt / demo; nhiều người thì chuyển sang hàng đợi job (RQ/Celery) |
| Demo không GPU | Video tổng hợp + detector theo màu (`detectors/demo.py`) | Chạy được trên máy bất kỳ, có sẵn các tình huống lỗi để trình diễn QA và lan truyền |

## Chưa làm

- Ẩn danh mặt/biển số (FR-03): EgoBlur cần tải weights có license, chưa tích hợp.
- Mask SAM2, VLM verifier, isotonic calibration (tuần 4 trong PLAN).
- Đăng nhập/phân vai (FR-21): hiện chỉ ghi tên người duyệt vào log.
