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
5. `uvicorn src.main:app` phục vụ API + UI. Người duyệt; mỗi thao tác ghi `corrections.jsonl`. Sau mỗi thao tác,
   QC kiểm lại nhãn cuối; approve bị chặn khi còn lỗi QC chưa sửa / chưa xác nhận.
6. Tab QC: audit ngẫu nhiên phần duyệt theo lô, checklist READY (`qc/audit.json`, `qc/qc_log.jsonl`).
7. `POST /api/v1/export` ghi dataset vào `data/workspace/exports/<id>/`: chỉ frame đã approve và sạch QC, kèm
   `qa_report.md` và manifest (READY / NOT_READY, SHA256 từng file).

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

## QC (`src/services/qc/`)

QA Agent chấm box detector lúc pipeline chạy; QC kiểm những gì đến *sau* đó.

```
               thao tác của người (sửa box, đổi lớp, vẽ, nhận box nội suy, duyệt theo lô)
                                  │
   checks.py  QC nhãn cuối ◄──────┘  chạy lại sau mỗi thao tác; chặn approve khi còn lỗi mở
                                  │
   audit.py   audit ngẫu nhiên ───┤  mẫu object BATCH_APPROVE + mẫu frame; Wilson 95%; mẫu sai -> mở lại frame
                                  │
   report.py  checklist READY ────┴─> exporter.py: bỏ frame còn lỗi QC, qa_report.md, manifest SHA256
   quick_check.py  file nhãn ngoài -> cùng checks.py + đối chiếu LiDAR / detection đã lưu, không ghi gì
```

| Check | Mức | Áp cho | Cách tính |
|---|---|---|---|
| `UNKNOWN_CLASS`, `INVALID_BOX`, `BOX_OUT_OF_IMAGE` | error | mọi nhãn | lớp ngoài taxonomy; cạnh < `qc.min_box_px` / toạ độ ngược / không hữu hạn; vượt biên ảnh > 1 px (đề xuất cắt về biên) |
| `NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH`, `ASPECT_RATIO_ABNORMAL`, `BOX_TOO_LARGE` | warning | nhãn cuối **khác** cái QA Agent đã chấm (box người vẽ, box đã sửa IoU < 0.99, lớp đã đổi) | gọi lại đúng `check_lidar` / `check_geometry` của QA Agent với box + lớp cuối, LiDAR lấy từ `workspace/lidar/` |
| `DUPLICATE_BOX` | warning | cặp cùng lớp IoU ≥ `qc.duplicate_iou` | đề xuất xoá box yếu hơn (ưu tiên giữ: người vẽ > người xem riêng > duyệt theo lô) |
| `OVERLAP_CROSS_CLASS` | warning | cặp khác lớp IoU ≥ `qc.cross_class_iou` | trừ cặp được phép (`pedestrian`+`bicycle`/`motorcycle`) |
| `AUDIT_MISSING_OBJECT` | warning | frame audit báo thiếu vật | gắn tay vào `frame.qc_manual`, xác nhận sau khi vẽ thêm box |
| `POSSIBLY_MISSING`, `MODEL_DISAGREES` | warning | chỉ Quick Check | detection keyframe score ≥ 0.5, có ở ≥ `min_support` sweep, không khớp nhãn nào (bỏ box người đã xoá); nhãn khớp detection IoU ≥ 0.5 mà khác lớp, score ≥ 0.6 |

| Quy tắc | Vì sao |
|---|---|
| Nhãn KEEP / duyệt theo lô mà không đổi thì không kiểm lại | Issue của nó người đã thấy lúc duyệt; báo lại chỉ làm ồn (đo: 0 finding trên output detector chưa sửa) |
| Error không xác nhận bỏ qua được, warning thì được (kèm lý do, ghi `qc_log.jsonl`) | Box hỏng không bao giờ là nhãn đúng; box trùng / tỉ lệ lạ đôi khi đúng thật |
| Xác nhận gắn với fingerprint (lớp + box cuối) | Sửa box sau khi xác nhận thì phải kiểm lại, không mang xác nhận cũ sang nhãn mới |
| Audit chỉ lấy object `BATCH_APPROVE` | Đó là phần duy nhất không ai xem riêng; tỉ lệ sửa nhóm low trong Metrics bằng 0 theo định nghĩa nên không đo được lỗi của nó |
| Đạt audit: cận trên Wilson 95% ≤ ngưỡng, hoặc đã audit hết tổng thể | Khoảng Wilson đúng cả khi 0 lỗi / mẫu nhỏ; audit hết thì không còn bất định lấy mẫu |
| Correction log phải khớp nhãn đang lưu (dòng log cuối của mỗi object) | Không có log nói đã sửa mà nhãn không đổi, không có nhãn đổi mà không có log |
| Quick Check tôn trọng quyết định file ghi lại (`qc_acknowledged`, `issues` của nhãn `KEEP`) | Export của chính hệ thống kiểm lại phải sạch; nhãn đã sửa box / đổi lớp thì issue cũ không còn áp dụng |

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Loader dữ liệu | Tự viết, chỉ numpy | nuscenes-devkit kéo nhiều dependency, pipeline chỉ cần vài bảng |
| Detector mặc định | YOLOE-26-L (open-vocab) | Một model phủ đủ 10 lớp bằng prompt, không cần ensemble. Số đo so với YOLO-World và các tổ hợp: [eval/results/detector_comparison.md](eval/results/detector_comparison.md) — mạnh hơn YOLO-World ở bus / barrier, yếu hơn ở traffic_cone / motorcycle |
| Fusion nhiều model | Chia điểm theo số model **nhận được lớp đó** | YOLO26 không có lớp cone: không để nó kéo điểm mọi cone xuống một nửa |
| Weights | `scripts/download_weights.py`, nhận file theo SHA256 của bản phát hành chính thức | File .pt là pickle (nạp là chạy code); bản trên Hugging Face openvision được kiểm là cùng model đã fuse |
| Grounding DINO | bản open-weight tiny | Bản 1.5 Edge chỉ có qua API |
| QA Agent | LangGraph, deterministic | Cần chạy trong batch và test; không có câu hỏi mở nào cần LLM |
| Lưu trữ | File JSON + JSONL | 404 frame, 1 người duyệt; không cần DB cho demo, dễ diff/export |
| UI | HTML/JS thuần do FastAPI phục vụ | Không cần build step, 1 lệnh là chạy |
| GT 2D | Hộp bao 8 đỉnh box 3D chiếu xuống | nuScenes không có box 2D gốc; báo cáo AP@0.5 là chính |
| Lan truyền 2D | Track qua sweep 12Hz bằng detection đã cache | Keyframe 2Hz quá thưa để khớp trực tiếp; không cần model mới (SAM2 video tốn GPU); `track_id` để gắn 3D sau |
| Video tải lên | Cắt thành frame + keyframe giống nuScenes | Dùng lại nguyên pipeline, QA Agent, review và lan truyền; không cần code riêng cho video |
| Xử lý video tải lên | `BackgroundTasks` trong process FastAPI | Đủ cho 1 người duyệt / demo; nhiều người thì chuyển sang hàng đợi job (RQ/Celery) |
| Demo không GPU | Video tổng hợp + detector theo màu (`detectors/demo.py`) | Chạy được trên máy bất kỳ, có sẵn các tình huống lỗi để trình diễn QA và lan truyền |
| QC nhãn cuối | Dùng lại hàm check của QA Agent, chạy trên nhãn cuối | Một định nghĩa cho "box đáng ngờ" ở cả pre-label, sau khi sửa và Quick Check |
| Chặn ở đâu | Approve frame (sửa ngay khi còn trên frame) + export (bỏ frame còn lỗi) | Lỗi phát hiện lúc approve rẻ nhất; export là chốt cuối, kể cả frame approve trước khi có QC |
| Lưu QC | `qc/audit.json` + `qc/qc_log.jsonl` chỉ ghi thêm, xác nhận nằm trong frame JSON | Cùng kiểu lưu file như phần còn lại; nhật ký tách khỏi `corrections.jsonl` để không lẫn vào đối chiếu log |

## Chưa làm

- Ẩn danh mặt/biển số (FR-03): EgoBlur cần tải weights có license, chưa tích hợp.
- Mask SAM2, VLM verifier, isotonic calibration (tuần 4 trong PLAN).
- Đăng nhập/phân vai (FR-21): hiện chỉ ghi tên người duyệt vào log.
