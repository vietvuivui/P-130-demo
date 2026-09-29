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

## 2026-09-29 (2) · Huy · nhánh `Huy`

**Làm gì:** Đổi detector mặc định từ YOLO-World sang **YOLOE-26-L** (https://huggingface.co/openvision/yoloe26-l-seg,
open-vocab, đủ 10 lớp bằng prompt). Thêm detector tuỳ chọn **YOLO26-L** (https://github.com/ultralytics/yolo26, COCO).
Fusion nhiều model chỉ tính phiếu của model nhận được lớp đó. Thêm script tải weights có kiểm SHA256. Đo trên 79
keyframe nuScenes: [eval/results/detector_comparison.md](eval/results/detector_comparison.md).

**File chính:**
- Mới: `src/services/detectors/yoloe26.py`, `src/services/detectors/yolo26.py`, `scripts/download_weights.py`,
  `tests/test_services/test_detectors_yolo26.py`, `eval/results/detector_comparison.md`.
- Sửa: `src/services/detectors/__init__.py` (`detector_classes`, `DetectorEnsemble.coverage`),
  `src/services/detectors/fusion.py` (tham số `coverage`), `src/models/qa_config.py`, `configs/autolabel.yaml`,
  `requirements-ml.txt`, `.gitignore` (`weights/`), README, ARCHITECTURE.

**Ảnh hưởng tới người khác:**
- **Config:** `detection.detectors` mặc định `[yoloe26]`; thêm mục `detection.yoloe26`, `detection.yolo26`.
  Muốn dùng lại model cũ: `detectors: [yolo_world]` hoặc `--detectors yolo_world` (cache cũ vẫn dùng được).
- **Weights mới** phải có trước khi `run`: `python scripts/download_weights.py` (≈ 330 MB vào `weights/`:
  yoloe-26l-seg.pt, mobileclip2_b.ts; `--all` thêm yolo26l.pt). Cần `ultralytics>=8.4`. File Hugging Face của
  openvision là cùng model đã fuse Conv+BN (đã kiểm pickle + so kết quả), script nhận nó làm nguồn dự phòng.
- `fuse_detections(per_model, iou, coverage=None)`: không truyền `coverage` thì hành vi như cũ.
- Workspace đang có vẫn là nhãn YOLO-World; frame "auto" gán nhãn lại bằng model mới:
  `python -m src.cli run --scenes <scene...> --overwrite`. Notebook Kaggle vẫn đặt `DETECTORS = ["yolo_world"]`.

**Cách kiểm tra:** `pytest` → 114 passed; `python scripts/download_weights.py --check`;
`python -m src.cli run --scenes scene-0061 --overwrite` rồi `python -m src.cli evaluate`.

**Còn dở / việc tiếp:**
- YOLOE-26-L một mình: mAP@0.5 0.281 so với 0.320 của YOLO-World — mạnh hơn ở bus / barrier, yếu hơn ở traffic_cone
  (0.151 so với 0.490), motorcycle, pedestrian.
- Tỉ lệ lỗi lọt trong nhóm low 0.459 (YOLO-World 0.277): chỉnh lại ngưỡng risk cho điểm số của model mới trước khi
  bật "Approve all low-risk".
- Chưa thử `imgsz` 640 cho YOLOE-26-L (model train ở 640; 1280 giúp vật nhỏ nhưng chưa đo).

## 2026-09-29 · Huy · nhánh `Huy`

**Làm gì:** Thêm phần QC gồm ba luồng. (1) **QC nhãn cuối**: sau mỗi thao tác của người, box người vẽ / đã sửa /
đổi lớp được kiểm lại hình học + LiDAR, cả frame được kiểm box trùng và box khác lớp chồng khít. Approve bị chặn khi
còn lỗi: sửa (có nút áp đề xuất) hoặc xác nhận cảnh báo kèm lý do. (2) **Audit ngẫu nhiên** phần duyệt theo lô (object)
và frame (vật bị sót), ước lượng tỉ lệ lỗi bằng khoảng Wilson 95%; mẫu sai thì frame mở lại. (3) **Quick Check** file
nhãn ngoài (COCO / JSONL), không ghi gì. Checklist READY trước khi xuất; export bỏ frame còn lỗi QC, thêm
`qa_report.md`, `qc_log.jsonl`, manifest có trạng thái READY + SHA256 từng file + commit.

**File chính:**
- Mới: `src/services/qc/` (`checks.py`, `quick_check.py`, `audit.py`, `report.py`),
  `tests/test_services/test_qc_*.py`, `tests/test_api/test_qc_api.py`.
- Sửa: `src/api/routes.py`, `src/services/exporter.py`, `src/services/store.py`, `src/models/schemas.py`,
  `src/models/qa_config.py`, `configs/autolabel.yaml`, `src/cli.py`, `src/web/*` (tab QC, khối QC trong panel review).

**Ảnh hưởng tới người khác:**
- **Approve đổi hành vi:** `POST /frames/{id}/approve` trả 409 `QC_FINDINGS` (kèm danh sách `findings`) khi nhãn cuối
  còn lỗi QC chưa xử lý. Tắt bằng `qc.gate_on_approve: false`. `review.approve_frame` (gọi thẳng service) không đổi.
- **Export đổi hành vi:** frame đã approve mà còn lỗi QC không được xuất (liệt kê trong `frames_skipped_qc`).
  `POST /export?require_ready=true` trả 409 `NOT_READY` khi checklist chưa đạt. File export thêm `qc_log.jsonl`,
  `qa_report.md`; `labels.jsonl` / `coco.json` thêm `qc_acknowledged` cho từng object, `labels.jsonl` thêm
  `width` / `height`.
- API mới: `GET /frames/{id}/qc`, `POST /frames/{id}/qc/ack`, `GET /qc/report`, `POST /qc/quick-check`,
  `GET /qc/audit`, `POST /qc/audit/sample`, `POST /qc/audit/{id}`. `/config` thêm mục `qc`.
- Schema: `FrameRecord` thêm `qc_acks`, `qc_manual` (mặc định rỗng, frame cũ đọc được); `IssueGroup` thêm `"qc"`
  (issue `AUDIT_FAILED`); `ExportResponse` thêm `release_status`, `frames_skipped_qc`; thêm `QCFinding`, `QCAck`,
  `AuditItem`, `QuickCheck*`.
- Config thêm mục `qc:` (ngưỡng box trùng, cặp lớp được phép chồng, Quick Check, cỡ mẫu + ngưỡng audit).
- Workspace thêm `qc/audit.json`, `qc/qc_log.jsonl`.
- 2 test cũ (`test_review_flow_and_export`, `test_export_only_approved`) được sửa theo hành vi mới: kịch bản đổi box
  100×30 px thành `traffic_cone` nay bị QC bắt `ASPECT_RATIO_ABNORMAL`, test xác nhận cảnh báo rồi mới approve / xuất.

**Cách kiểm tra:**
- `pytest` → 110 passed (82 cũ + 28 mới); `ruff check src tests` sạch.
- `python -m src.cli qc-report` (exit 1 nếu chưa READY); `python -m src.cli quick-check <file.json>` (exit 1 nếu có lỗi).
- UI: sửa một box thành dẹt / vẽ trùng lên box có sẵn → khối "QC Nhãn cuối" hiện ở đầu panel, nút approve khoá;
  tab QC → lấy mẫu audit, bấm `Y` / `X`; Quick Check một file `coco.json` đã xuất → 0 lỗi mở.
- Đã chạy E2E trên bản sao workspace nuScenes thật (79 keyframe) bằng Edge headless: chặn approve → áp đề xuất →
  xác nhận cảnh báo → audit → READY → xuất.

**Còn dở / việc tiếp:**
- Ngưỡng `POSSIBLY_MISSING` (score 0.5) chưa có số đo precision (thiếu GT "vật bị sót" thật); trên GT nuScenes báo
  ~0.6 gợi ý / frame.
- Audit chưa phân tầng theo lớp / scene; cỡ mẫu mặc định 50 object, 10 frame.
- Quick Check chưa nhận KITTI `.txt` (dữ liệu dự án là nuScenes); thêm parser ở `quick_check.parse_labels` nếu cần.

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
