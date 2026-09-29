# CHANGELOG — nhóm P-130 (AutoLabel 3D)

Mỗi lần **push**, người push thêm **một mục lên đầu** file này (mục mới nhất ở trên cùng). Mục đích: ai mở repo
cũng biết nhánh nào đang có gì, ai sửa file nào, cần chạy gì để kiểm tra — để 4 người cùng làm mà repo không loạn.

`WORKLOG.md` (deliverable #9 của BTC) vẫn ghi theo ngày, mỗi người một dòng tóm tắt. File này là bản chi tiết.

## Cách ghi

```markdown
## YYYY-MM-DD · Tên · nhánh `ten-nhanh`

**Làm gì:** 1–3 câu, nói kết quả chứ không kể quá trình.

**File chính:** file mới / file sửa quan trọng (không cần liệt kê hết).

**Ảnh hưởng tới người khác:** schema, API, config, tên hàm dùng chung đã đổi; ghi "Không" nếu không có.

**Cách kiểm tra:** lệnh chạy được ngay (`pytest`, lệnh CLI, thao tác UI).

**Còn dở / việc tiếp:** những gì chưa xong, ai nên làm tiếp.
```

Quy tắc:

- Một lần push = một mục. Push nhiều commit nhỏ trong cùng một việc thì gộp vào một mục.
- Đổi schema (`src/models/schemas.py`), config (`configs/autolabel.yaml`) hay API thì **bắt buộc** ghi ở
  "Ảnh hưởng tới người khác", vì đó là chỗ làm hỏng code của người khác mà không ai biết.
- Trước khi push: `ruff check src tests` và `pytest` phải sạch; ghi số test pass vào "Cách kiểm tra".

---

## 2026-09-29 · Danh · nhánh `danh`

**Làm gì:** Tích hợp bộ giao diện UI Flow đa trang hoàn chỉnh (Đăng nhập, Dự án, Chọn frame, Xuất dữ liệu, Sơ đồ Flow) kết nối liền mạch vào luồng chạy chính của ứng dụng qua các route FastAPI. Đồng bộ toàn diện hệ thống thiết kế và bảng màu trang Workspace (Workstation gán nhãn) theo chuẩn Navy Slate (`#0f172a`) và Light Slate (`#f8fafc`), loại bỏ lỗi lệch màu do dark mode tự động, chuẩn hóa màu sắc bounding box canvas và thẻ QA risk badge.

**File chính:**
- Mới: `src/web/login.html`, `src/web/projects.html`, `src/web/frames.html`, `src/web/export.html`, `src/web/flow.html`, `src/web/ui-flow.css`, `src/web/ui-flow.js`, `src/web/road_camera.jpg`.
- Sửa: `src/main.py` (thêm route chuyển hướng giao diện `/login`, `/projects`, `/frames`, `/annotate`, `/export`, `/flow`), `src/web/index.html` (thanh 7-step stepper, topbar brand đồng bộ icon/link), `src/web/styles.css` (bỏ prefers-color-scheme dark, chuẩn hóa token màu, topbar navy slate, layout flexbox), `src/web/app.js` (đồng bộ màu vẽ bounding box `RISK_COLOR` & `HUMAN_COLOR`).

**Ảnh hưởng tới người khác:** Route mặc định `/` chuyển hướng về `/ui/login.html` thay vì `/ui/`. Thêm các đường dẫn trang `/projects`, `/frames`, `/export`, `/flow`. Các API `/api/v1/*`, schema dữ liệu và cấu hình `configs/autolabel.yaml` giữ nguyên vẹn.

**Cách kiểm tra:**
- `ruff check src tests` → All checks passed.
- `python -m pytest tests/test_api/` → 11 passed.
- Chạy `python -m src.demo` hoặc `uvicorn src.main:app`, truy cập `http://localhost:8000/`:
  - Kiểm tra luồng Đăng nhập (`/login`) → Danh sách dự án (`/projects`) → Chọn frame (`/frames`) → Bấm "Gán nhãn" vào Workspace (`/annotate` hoặc `index.html`).
  - Kiểm tra đồng bộ tông màu: thanh 7 bước, topbar `#0f172a`, canvas stage, thẻ rủi ro High/Medium/Low, phím tắt và thao tác duyệt frame/video hoạt động bình thường.
  - Xem toàn cảnh flow tại `/flow` và trang xuất tại `/export`.

**Còn dở / việc tiếp:**
- Bổ sung xác thực JWT thực tế cho `/login` khi triển khai production nhiều annotator.

---

## 2026-09-28 · Việt · nhánh `viet`

**Làm gì:** Notebook Colab chạy liền một mạch: PointPillars (MMDetection3D, 10 sweep) tạo box 3D trên `mini_val` và chấm
mAP/NDS bằng `DetectionEval`, rồi YOLOE-26 (open-vocabulary, đặt đủ 10 lớp nuScenes bằng prompt) kiểm chứng từng box
trên camera nhiều thông tin nhất. Mỗi box nhận một kết luận (DUNG / DUNG VAT, BOX LECH / SAI LOP / NGHI BAO NHAM /
CAMERA KHONG XAC NHAN / CHUA DU THONG TIN) kèm nhận xét; nhãn gốc chỉ in kèm để đối chiếu.

**File chính:** Mới: `demo/model.ipynb`, `scripts/verify_objects.py` (bộ kiểm chứng; notebook ghi lại nguyên văn file
này ở Cell 8, nên chạy local hay Colab đều dùng chung code).

**Ảnh hưởng tới người khác:** Không. Không đụng `src/`, schema, config hay API.

**Cách kiểm tra:** Colab GPU T4, có `v1.0-mini.tgz` trong Drive: chạy `demo/model.ipynb` từ Cell 1 đến 15. Local:
`NUSC_DATAROOT=<thu muc nuScenes> python scripts/verify_objects.py --yoloe yoloe-26s-seg.pt` (không có `--results` thì
dùng box mô phỏng). Không sửa `src/`/`tests/` nên chưa chạy `pytest`.

**Còn dở / việc tiếp:** Chỉnh `PROMPTS` / `CLASS_CONF` theo bảng các cặp lệch lớp (Cell 13); `mini_val` chỉ có 2 scene
nên các tỷ lệ chưa đủ để kết luận thống kê.

## 2026-09-27 (3) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Zoom ảnh trên khung chỉnh sửa (lăn chuột quanh con trỏ, `+`/`−`/`0`, nút `− 100% +`, kéo để di chuyển
khi phóng to), dùng cho cả chế độ Ảnh và Video. Thêm chú thích box GT (tooltip ô GT, README). Script đóng gói nuScenes
chọn ngẫu nhiên scene có trên máy (`--random`, `--window`) và đọc bảng trainval kiểu stream; video thưa (chỉ keyframe)
bỏ check temporal thay vì báo FLICKER sai. Test trên 2 scene trainval ngẫu nhiên (scene-0031, scene-0065).

**File chính:** `src/web/app.js`, `src/web/index.html`, `src/web/styles.css`, `scripts/pack_nuscenes_subset.py`,
`src/services/video.py`, `src/models/qa_config.py` (`video.max_sweep_gap_s`), `configs/autolabel.yaml`, README.

**Ảnh hưởng tới người khác:** Config thêm `video.max_sweep_gap_s`. Canvas nằm trong `#canvas-scroll`; phím `+ - = 0`
dành cho zoom.

**Cách kiểm tra:** `pytest` → 82 passed. Mở UI, lăn chuột trên ảnh, vẽ box khi đang zoom, bấm `0`.

**Còn dở / việc tiếp:** Số đo trên trainval chạy YOLO-World bằng từ vựng COCO trên CPU; chạy lại trên GPU với đủ prompt.

## 2026-09-27 (2) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Tách lan truyền nhãn thành chế độ riêng. Thanh trên cùng có hai lựa chọn: **🖼 Ảnh** (MVP của Huy, duyệt
từng frame theo risk, không lan truyền) và **🎞 Video** (danh sách video + timeline riêng; bấm hoặc kéo một frame từ
timeline thả lên khung ảnh để sửa như 2D; approve thì lan truyền sang các frame sau). Nguồn video: scene nuScenes
hoặc **mp4 tải lên** (cắt 10 fps, keyframe 2 fps, auto-label + QA chạy nền). Thêm **demo không cần GPU/nuScenes**
(`python -m src.demo`), kịch bản trình diễn, và notebook Kaggle + script đóng gói để chạy nuScenes thật khi máy chưa
tải được dataset.

**File chính:**
- Mới: `src/services/video.py` (cắt mp4, xử lý nền, danh sách/timeline video), `src/services/detectors/demo.py`
  (detector theo màu cho demo/test), `src/demo.py`, `scripts/pack_nuscenes_subset.py`,
  `demo/kaggle_nuscenes_run.ipynb`, `docs/demo-script.md`, `tests/test_services/test_video.py`,
  `tests/test_services/test_pack_subset.py`, `tests/test_api/test_video_api.py`.
- Sửa: `src/web/*` (chế độ Ảnh/Video, timeline, kéo-thả, phát, tải lên), `src/api/routes.py`,
  `src/services/pipeline.py` (tách `label_keyframe` dùng chung), `src/services/sequence.py`
  (`WorkspaceSequenceSource`), `src/services/store.py`, `src/services/nuscenes_data.py`, `src/services/exporter.py`,
  `src/cli.py`, `src/models/*`, `configs/autolabel.yaml`, `requirements.txt`, `Makefile`, README, ARCHITECTURE.

**Ảnh hưởng tới người khác:**
- API mới: `GET /api/v1/videos`, `GET /api/v1/videos/{id}`, `POST /api/v1/videos/upload` (multipart `file`).
  `POST /frames/{id}/propagate` giờ chạy được cả trên video tải lên; scene nuScenes mà máy không có bảng
  nuScenes thì trả 503 `DATA_UNAVAILABLE`.
- Schema: thêm `TimelineEntry`, `VideoRecord`, `VideoSummary`, `VideoDetail`, `VideoFrame`. Frame của video tải
  lên có `camera="video"`, `scene=<video_id>`, đường dẫn ảnh dạng `@workspace/videos/<id>/frames/NNNNN.jpg`
  (`WorkspaceStore.resolve` đổi ra đường dẫn thật).
- Config: thêm mục `video:` (track_fps, label_fps, max_seconds, max_upload_mb) và `detection.demo`.
- `requirements.txt` thêm `opencv-python-headless`, `python-multipart` → chạy lại `pip install -r requirements.txt`.
- `pipeline.AutoLabelPipeline.process` giờ gọi hàm `label_keyframe(...)` (cùng kết quả); ai sửa pipeline thì
  sửa ở `label_keyframe`.
- `src/main.py`: lúc khởi động chạy nền `resume_videos()` (làm tiếp video tải lên còn dở). `DetectorEnsemble._get`
  có khoá; `WorkspaceStore._write_json` thử lại khi Windows báo file đang bị đọc.
- Export: `file_name` của ảnh video tải lên bỏ tiền tố `@workspace/`, thêm trường `image_root`
  (`nuscenes` / `workspace`). `evaluate` chỉ tính frame có GT (bỏ qua video tải lên).
- UI: chế độ Ảnh giữ nguyên hành vi MVP (approve xong mở frame rủi ro cao nhất, không lan truyền). Nút lan truyền
  và ô "Lan truyền khi approve" chuyển xuống thanh timeline của chế độ Video. Phím mới: `Space` phát/dừng video.

**Cách kiểm tra:**
- `pytest` → 80 passed (65 cũ + 15 mới); `ruff check src tests` sạch.
- `python -m src.demo` → http://localhost:8000 → 🎞 Video → làm theo `docs/demo-script.md` (đã chạy thử đầu cuối
  bằng Playwright: đổi lớp + xoá ở frame #1, approve → 20 frame nhận nhãn, kéo frame #9 lên khung ảnh, phát, tải lên mp4).
- nuScenes thật: `python -m src.cli run --scenes scene-0061` rồi mở UI, chế độ Video. Hoặc chạy
  `demo/kaggle_nuscenes_run.ipynb` trên Kaggle, tải zip về, giải nén, sửa `.env` theo README trong zip.

**Còn dở / việc tiếp:**
- Notebook Kaggle mới chạy thử phần không cần GPU (tìm dataset, dựng dataroot, đóng gói) trên dataset giả; cần chạy
  thật một lần trên Kaggle để xác nhận bước `run`/`detect-sweeps` với YOLO-World.
- Video tải lên xử lý bằng `BackgroundTasks` trong process server, mỗi lúc một video (khoá chung vì detector không
  chạy song song được). Server tắt / `--reload` giữa chừng thì lần khởi động sau tự làm tiếp. Nhiều người dùng thì nên
  chuyển sang hàng đợi job.
- Vật bị che hẳn lâu hơn `max_coast_images` thì track dừng, hiện lại là object mới.

---

## 2026-09-27 · Kiên · nhánh `feat/label-propagation` (tách từ `Huy`)

**Làm gì:** Thêm lan truyền nhãn 2D trên video (FR-11 → FR-13 trong PRD). Approve một keyframe thì mọi quyết
định của người (giữ, sửa box, đổi lớp, vẽ thêm, xoá) được mang sang các keyframe sau của scene: tracker đi qua mọi
ảnh CAM_FRONT 12Hz giữa các keyframe, dùng detection YOLO-World đã cache, chỉ ghi vào keyframe chưa ai mở và
dừng trước frame người đã mở. Mỗi nhãn lan truyền có độ tin cậy `c_prop`; c_prop thấp hoặc detector không thấy
thì bị gắn issue để người xem kỹ, còn lại rơi vào nhóm low để duyệt theo lô. Box người đã xoá ở keyframe được tự
xoá ở frame sau (khôi phục được bằng Keep). Có thêm thí nghiệm "keyframe hoàn hảo" đo lan truyền so với GT mà
không cần người gán.

**File chính:**
- Mới: `src/services/propagation.py` (tracker, c_prop, ghi vào frame), `src/services/sequence.py` (chạy cả scene,
  thí nghiệm GT), `tests/test_services/test_propagation.py`, `tests/test_services/test_sequence_data.py`,
  `tests/test_api/test_propagation_api.py`, `CHANGELOG.md`.
- Sửa: `src/models/schemas.py`, `src/models/qa_config.py`, `configs/autolabel.yaml`, `src/services/nuscenes_data.py`
  (`camera_timeline`), `src/services/detectors/__init__.py` (`load_cached`), `src/services/review.py` (metrics),
  `src/services/evaluation.py`, `src/services/exporter.py`, `src/api/routes.py`, `src/cli.py`, `src/web/*`,
  `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `WORKLOG.md`, `.github/PULL_REQUEST_TEMPLATE.md`.

**Ảnh hưởng tới người khác:**
- Schema: `LabelObject.source` thêm giá trị `"propagated"`; `LabelObject` thêm `track_id`, `propagation`;
  `FrameRecord` thêm `propagated_from`, `propagated_at`, `prelabel` (bản pre-label trước lần lan truyền đầu);
  `FrameSummary` thêm `propagated_from`; `ReviewState.action` thêm `"PROPAGATED_DELETE"`. Tất cả có giá trị mặc
  định nên file frame JSON cũ vẫn đọc được.
- Config: thêm mục `propagation:` trong `configs/autolabel.yaml` (ngưỡng + trọng số c_prop).
- API mới: `POST /api/v1/frames/{id}/propagate`. `GET /metrics` thêm khối `propagation`; `m4_correction_rate`
  vẫn chỉ tính object detector sinh, nhãn lan truyền có M4 riêng; `flag_recall` giờ tính cả nhãn lan truyền.
- Export: mỗi object thêm `track_id`, `propagated_from`.
- Đánh giá pre-label (`evaluate`) dùng `frame.prelabel` nếu frame đã được lan truyền.
- UI: nút "↦ Lan truyền" (phím `T`), ô "Lan truyền khi approve" (mặc định bật). Approve xong UI mở keyframe kế
  tiếp của cùng scene thay vì frame rủi ro cao nhất.

**Cách kiểm tra:**
- `pytest` → 65 passed (48 cũ + 17 mới); `ruff check src tests` sạch.
- Với dữ liệu thật: `python -m src.cli run --scenes scene-0061` → mở UI, duyệt `scene-0061_000`, bấm Approve →
  `scene-0061_001` mở ra với nhãn `↦ c …`.
- `python -m src.cli detect-sweeps --scenes scene-0061` (GPU) để tracker có detection ở mọi ảnh 12Hz, rồi
  `python -m src.cli eval-propagation` → `eval/results/propagation_eval.md`.

**Còn dở / việc tiếp:**
- Chưa chạy trên nuScenes thật (máy của Kiên chưa có dataset lúc làm): cần chạy `eval-propagation` rồi chỉnh
  `propagation.weights` / `flag_below` theo số thật. Đã thử bằng mô phỏng (detection rung ±4 px, rơi 15%, hai xe cắt
  nhau): 10 keyframe không đổi ID, IoU trung bình ~0.88.
- Chỉ 2D trên CAM_FRONT. `track_id` đã có để gắn box 3D (P2) vào sau.
- Mask SAM2 chưa có trong MVP nên lan truyền mới mang box, chưa mang mask.

---

## 2026-09-26 · Huy · nhánh `Huy` — ghi bù từ commit `454bb89`

**Làm gì:** Thay agent mẫu của template bằng pipeline auto-label 2D: detector open-vocab (YOLO-World, Grounding
DINO, Florence-2) có fusion, QA Agent LangGraph (confidence, LiDAR, temporal, issue, risk), nạp nuScenes, review
by exception, xuất dataset, CLI, UI web, test.

**File chính:** `src/services/*`, `src/agents/nodes/*`, `src/web/*`, `src/cli.py`, `configs/autolabel.yaml`,
`PLAN-2D-AutoLabel-VLM-Agent.md`.

**Ảnh hưởng tới người khác:** Thay toàn bộ code mẫu của template (`example_node`, `example_tool`, `llm.py` đã xoá).

**Cách kiểm tra:** `pytest` → 48 passed; `python -m src.cli run --limit 40` rồi `uvicorn src.main:app`.

**Còn dở / việc tiếp:** Ẩn danh (EgoBlur), mask SAM2, VLM verifier, đăng nhập/phân vai (theo README).
