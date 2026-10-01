# Architecture — AutoLabel 3D (2D + 3D)

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

## UI: ba chế độ

| | 🖼 Ảnh | 🎞 Video |
|---|---|---|
| Danh sách trái | Hàng đợi frame theo risk (MVP) | Video: scene nuScenes (suy từ frame) + mp4 tải lên (`GET /videos`) |
| Dưới khung ảnh | Dải temporal t-2 … t+2 | Timeline của video (`GET /videos/{id}`): kéo thẻ thả lên khung ảnh để mở |
| Approve xong | Mở frame rủi ro cao nhất tiếp theo | Lan truyền (nếu bật) rồi mở frame kế tiếp chưa duyệt của video |
| Lan truyền | Không | Tự chạy khi approve, hoặc nút ↦ / phím `T` |

Khung chỉnh sửa (canvas + panel risk) là một, dùng chung cho cả hai chế độ.

Chế độ 🧊 3D (`src/web/app3d.js`, ES module, three.js vendored ở `src/web/vendor/`) có panel riêng: hàng đợi frame 3D
theo mô hình, khung 3D / BEV, ảnh camera có box chiếu xuống, card từng box với kết luận kiểm chứng. `app.js` phát sự kiện
`autolabel:mode` / `autolabel:tab` để hai phần không phụ thuộc nhau.

## Dự án của end-user (web)

1. `POST /api/v1/projects` (multipart) lưu file vào `data/projects/<id>/upload/`, xếp dự án vào hàng đợi.
2. `ProjectManager` (`src/services/projects.py`) chạy một luồng nền, xử lý từng dự án, từng bước, dưới cùng khoá
   GPU với video tải lên (`video._PROCESS_LOCK`). Tiến độ ghi vào `project.json`, UI hỏi lại mỗi 2 s. Các bước:
   - **ingest**: giải nén an toàn (chặn zip-slip), nhận dạng loại (`ingest.detect_kind`). Video / ảnh thành record
     video như mp4 tải lên. KITTI và "LiDAR + camera" đổi sang bảng nuScenes (`ingest/nusc_writer.py`): LiDAR lưu
     theo trục nuScenes, cường độ 0–255, calib camera, ego pose (oxts / `poses.json`), sweep = frame trước.
     Zip nuScenes giữ nguyên.
   - **label2d**: pipeline 2D như nuScenes (CAM_FRONT) hoặc video.
   - **predict3d**: tiến trình con `.venv-mm3d/python tools3d/run3d.py predict`, tiến độ đọc từ log mmengine.
     Không có môi trường 3D thì bỏ qua và ghi lý do.
   - **verify3d**: `run_label3d` với ngưỡng điểm tự chọn theo F1 của mô hình (`auto_min_score`).
3. Mỗi dự án có `WorkspaceStore` và dataroot riêng. Router review 2D / 3D được gắn thêm lần nữa ở
   `/p/{project_id}/api/v1`; dependency `get_store` / `data_source` đọc `project_id` trong đường dẫn. Nhờ vậy UI review
   dùng nguyên, chỉ đổi tiền tố API theo `?project=`.
4. Xuất nuScenes (`export_nusc.py`): box đã duyệt thành `sample_annotation` (đổi sang hệ toàn cục, đếm điểm LiDAR);
   box cùng `track_id` nối thành một `instance` (`first/last_annotation_token`, `prev/next`). Kèm bảng gốc của dataset.

## Phần 3D

1. `tools3d/run3d.py` (môi trường `.venv-mm3d`, GPU): tìm scene val có đủ file, tạo info MMDet3D, suy luận các mô hình
   có trọng số nuScenes, chấm `DetectionEval` chỉ trên các sample đã chạy → `eval/results/det3d/summary.json`,
   `<model>/ui_preds.json` (3 scene demo).
   `eval` thêm `ensemble`: gộp 4 mô hình LiDAR theo khoảng cách tâm từng lớp (`refine3d.fuse`), rồi tinh chỉnh theo
   track (`refine3d.refine_tracks`: kích thước trung vị, vận tốc sai phân có chốt an toàn, sửa hướng lật, nội suy
   keyframe sót). `--tta` chạy thêm 3 lượt lật trục (thay `MultiScaleFlipAug3D` bằng `RandomFlip3D` cố định, lật ngược
   kết quả trong `test_step`). Số liệu tách dev (3 scene demo) / held-out.
2. `python -m src.cli label3d --model M` (môi trường chính): mỗi keyframe gộp LiDAR keyframe + 4 sweep (bù ego-motion),
   đổi box sang hệ LiDAR, chạy YOLOE trên 6 camera (dùng lại cache detection), `verify3d.verify_boxes` kết luận từng box.
   Ghi `frames3d/<model>/<frame>.json`, `lidar3d/<frame>.bin` (float32 xyzi, tối đa 60k điểm), `gt3d/`.
3. `routes3d.py` phục vụ UI; thao tác ghi `corrections3d.jsonl`; export ra box hệ toàn cục định dạng nuScenes detection.
4. `python -m src.cli evaluate3d` so kết luận kiểm chứng với GT (khớp tâm < 2 m).
5. Người vẽ thêm box (`ADD_BOX`, `source: human`) cho vật mô hình bỏ sót và sửa box (`EDIT_BOX`, box mô hình giữ ở
   `original_box`). Việc đặt lên mặt đường / co khít điểm LiDAR chạy ngay trên trình duyệt với point cloud đã tải.
6. Ảnh BEV (`bev.py`): độ cao mặt đường z0 = mode của z các điểm LiDAR thấp quanh xe; mỗi camera có homography
   H = K·[r1 r2 z0·r3+t] từ mặt đường sang ảnh (mặt đường thật lấy từ LiDAR: RANSAC + lưới sai lệch). Ô mà tia camera bị
   vật cao chắn không lấy màu từ camera đó. Ghép thêm ±4 keyframe cùng scene: ô mặt đường đổi sang hệ của từng keyframe
   bằng `global_from_lidar`, mỗi lần nhìn chấm theo độ nét (1/(1+(ρ/8)²)⁶, ρ = khoảng cách tới camera) nên lần nhìn gần
   thắng hẳn; bỏ vùng quanh thân xe (camera thấy cốp / capô của chính xe), dọn mảnh vụn, tô lỗ < 3 m². Cache PNG theo
   frame ở `workspace/bev3d/`.
7. BEV của chế độ Ảnh / Video (`bev2d.js`): một camera; ảnh nuScenes dùng LiDAR và ghép keyframe lân cận (ego pose) như
   ảnh BEV 3D, video tải lên dùng mặt đường z = 0 của hệ ego; `/frames/{id}/bev/meta` trả
   cam_from_ego + homography, trình duyệt đổi điểm LiDAR (u, v, độ sâu) về hệ ego và đặt box 2D lên mặt đường.

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
   ở ảnh kế tiếp bằng **optical flow** giữa hai ảnh (`src/services/flow.py`, OpenCV DIS; `propagation.flow`:
   `always` mặc định, `missing` chỉ ở ảnh chưa có detection, `off` = vận tốc không đổi như trước), ghép với
   detection đã cache theo hai lượt kiểu ByteTrack (`propagation.association: byte`): box score ≥ 0.3 trước
   (IoU ≥ 0.3), box score thấp chỉ cho track còn thiếu (IoU ≥ 0.5), một-một. Sweep người đã sửa (xem dưới) thay cho cache detector. Ảnh chưa có
   detection: box đi theo flow. Dừng track khi > 6 ảnh liền không khớp, ra khỏi khung, hoặc box quá nhỏ.
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

## Lan truyền box 3D

`src/services/propagation3d.py`: approve keyframe 3D → track từ box đã duyệt (lớp / kích thước người chốt; box người
xoá → track "suppress"). Mỗi keyframe sau: box về hệ toàn cục qua `global_from_lidar`, dịch `c + v·dt` (vận tốc của
mô hình, vật đứng yên thì không), khớp dự đoán mô hình theo khoảng cách tâm BEV ≤ ngưỡng theo lớp của tracker
CenterPoint × `propagation3d.dist_scale`; khớp → object `source="propagated"` (tâm / hướng từ mô hình, lớp / kích thước
từ người, mức low nếu khớp chặt và cùng lớp), không khớp → track chạy tiếp, mất > `max_misses` (4) keyframe thì dừng. Như 2D:
dựng lại từ `prelabel`, không ghi vào frame đã mở, box người xoá chỉ tự xoá box cùng lớp (`PROPAGATED_DELETE`).
Đánh giá: `src/services/propagation3d_eval.py`, `tools3d/eval_propagation3d.py`.

OC-SORT (`propagation.oc_*`, `propagation3d.oc_*`, tắt mặc định): OCR ghép thêm track đang mất theo box / tâm quan sát
cuối, ORU lấy vận tốc theo quan sát cuối → quan sát mới sau khi mất, OCM ưu tiên detection cùng hướng đi đã quan sát. Đo
trước / sau: `eval/results/ocsort.md` — 2D không lợi; 3D chỉ phần giữ track lâu hơn (`max_misses` 2 → 4) được bật.

## Optical flow và sửa nhãn ở sweep

Ý tưởng lấy từ *Deep Feature Flow* (Zhu et al., arXiv:1611.07715): tính kỹ ở keyframe, mang kết quả sang ảnh lân cận
bằng flow thay vì chạy lại detector hay đoán. Bài báo warp feature map bên trong mạng (phải train lại cả mạng với
FlowNet); ở đây làm ở mức box nên không phải train và không đổi detector:

- `flow.FlowField.warp_box`: dời box theo từng cạnh (trung vị flow ở dải trái / phải / trên / dưới bên trong box),
  theo được cả vật tiến lại gần. DIS ở nửa độ phân giải: ~0.06 s mỗi cặp ảnh trên CPU 2 nhân.
- **Lan truyền** (`Tracker`, trên): bật mặc định.
- **Check temporal** (`qa.temporal.flow`) và **tính lại score theo sweep** (`qa.temporal.rescore`,
  `temporal_fusion.py`, ý tưởng Flow-Guided Feature Aggregation ở mức box): có sẵn, tắt mặc định — đo trên dev
  không tăng mAP và làm lỗi lọt qua duyệt theo lô tăng (eval/results/temporal/report.md).
- **Sửa tự do ở sweep** (`sweep_review.py`, `POST /frames/{id}/sweeps/{offset}/actions`): keep / xoá / đổi lớp /
  sửa box / vẽ thêm / khôi phục. Bản máy (`SweepInfo.detections`) giữ nguyên, bản người sửa ở `SweepInfo.boxes`.
  Mỗi thao tác tính lại keyframe (`requalify_frame`): score (nếu bật rescore), FLICKER, đề xuất RECOVERED_BY_TRACK
  (đề xuất cũ còn chờ được thay, cái người đã xử lý giữ nguyên), risk; kết quả LiDAR lúc auto-label giữ nguyên.
  Sweep không được xuất. Có undo/redo và sự kiện `sweep_action` trong `events.jsonl`.

Đánh giá trước / sau: `python tools2d/eval_temporal.py --dataroot … --workspace … --out …` (dùng lại cache detection
của workspace, mọi cấu hình cùng một detection).

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
| Mô hình 3D | Trọng số MMDetection3D có sẵn, chạy trong venv riêng | Không cần train; stack mmcv/torch cũ không trộn với môi trường chính (torch mới cho YOLOE) |
| Kiểm chứng 3D | Chiếu box xuống 6 camera + detector 2D + điểm LiDAR, không dùng GT | Port bộ `verify_objects.py` của nhóm 3D; tách được "không thấy vì bị che/tối" khỏi "báo nhầm" |
| Viewer 3D | three.js vendored, không build step | Giữ nguyên nguyên tắc UI 1 lệnh là chạy, chạy offline |
| Ảnh BEV | Homography mặt đường (IPM), không dùng mô hình | Không cần GPU hay train; đủ cho mục đích gán nhãn (vạch đường, vị trí trên làn). Vật cao bị kéo nhoè — mô hình BEV học được (LSS/BEVFormer) mới sửa được, không đáng cho công cụ gán nhãn |
| Tăng độ chính xác 3D | Ensemble 4 mô hình LiDAR + tinh chỉnh theo track (offboard) + TTA lật, không train | Dev mAP 0.537 → 0.644. Gán nhãn được nhìn cả tương lai nên dùng được track hai chiều; mô hình camera làm ensemble tệ đi, chấm lại điểm bằng camera làm mAP giảm (kiểm tra chéo) nên không dùng |
| Tăng độ chính xác 2D | Fine-tune YOLOE trên nuImages (`tools2d/`) thay vì dò prompt | Đổi prompt / prompt gây nhiễu / lật ảnh không tăng mAP trên dev; nuImages cùng miền, loại ảnh trùng log val để không rò rỉ, chấm quyết định trên held-out |
| Web end-user | Dự án = thư mục riêng + một luồng nền, router review gắn lại dưới `/p/{id}` | Một GPU, không đăng nhập (theo yêu cầu); dùng lại toàn bộ UI review, không nhân bản code |
| Định dạng trung gian | Mọi dữ liệu có LiDAR đổi sang bảng nuScenes | Loader, mô hình 3D, kiểm chứng, xuất đều đã viết cho nuScenes; thêm định dạng mới chỉ cần một converter |
| Demo không GPU | Video tổng hợp + detector theo màu (`detectors/demo.py`) | Chạy được trên máy bất kỳ, có sẵn các tình huống lỗi để trình diễn QA và lan truyền |

## Duyệt: trả lại, hoàn tác, lịch sử, năng suất

- Mỗi thao tác sửa nhãn đẩy bản frame trước đó vào `workspace/history/<2d|3d-model>/<frame>.json` (50 bước); undo / redo
  đổi chỗ bản hiện tại với bản trong ngăn, ghi log `UNDO` / `REDO`. Frame đã approve phải mở lại trước khi hoàn tác.
- Reject (lý do bắt buộc) đặt `status = "rejected"`: không xuất được, lên đầu hàng đợi; sửa lại thì về `editing` nhưng giữ lý do.
- `events.jsonl` ghi approve / reject / reopen / undo / redo; năng suất (FR-27) tính từ frame đã approve (người, thời gian
  duyệt) và `autolabel_s` / `autolabel_run` mà pipeline ghi cho từng frame.

## Chưa làm

- Ẩn danh mặt/biển số (FR-03) đã có bằng detector mở (`privacy.py`); EgoBlur (chính xác hơn) cần weights có license, chưa tích hợp. Ảnh BEV chưa làm mờ.
- Mask SAM2, VLM verifier, isotonic calibration (tuần 4 trong PLAN).
- Đăng nhập/phân vai (FR-21): hiện chỉ ghi tên người duyệt vào log; web dự án dành cho một máy chủ, một nhóm.
