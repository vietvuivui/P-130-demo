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

## 2026-09-29 (3) · Kiên · nhánh `kien`

**Làm gì:**

- **Web cho end-user (trang Dự án, mặc định khi mở `/`):**
  - tải lên video, ảnh, zip nuScenes, KITTI, hoặc "LiDAR rời + camera";
  - xử lý nền, có tiến độ từng bước;
  - duyệt 2D / 3D bằng UI review có sẵn, trên đúng dự án;
  - xuất nuScenes (sample_annotation + instance nối theo track) hoặc COCO.
- **3D:**
  - ensemble 4 mô hình LiDAR + tinh chỉnh theo track (offboard);
  - TTA lật trục;
  - `run3d.py predict` cho dữ liệu chưa gán nhãn.
  - Trên 3 scene dev: mAP 0.537 → 0.644, NDS 0.560 → 0.643 so với CenterPoint voxel.
- **2D:** gói fine-tune YOLOE-26L trên nuImages (`tools2d/`), gồm converter chống rò rỉ, huấn luyện linear probe /
  full, chấm dev / held-out, và hướng dẫn chuyển file giữa hai máy không chung LAN. Đổi prompt / thêm prompt gây nhiễu
  / lật ảnh / 1600 px đã đo và không tăng mAP (`eval/results/det2d_variants.json`).

**File chính:**

- mới: `src/services/projects.py`, `src/services/ingest/` (KITTI, LiDAR+camera, ghi bảng nuScenes),
  `src/services/export_nusc.py`, `src/api/projects_routes.py`, `src/web/projects.{html,js}`, `tools3d/refine3d.py`,
  `tools2d/`;
- sửa: `tools3d/run3d.py` (eval ensemble + dev/held-out, `--tta`, `predict`), `src/api/routes.py` (store / dataroot
  theo dự án), `src/main.py`, `src/web/app.js` / `app3d.js` / `bev2d.js` (`?project=`),
  `src/services/detectors/yoloe.py` (`tta_flip`, nhận trọng số đã fine-tune).

**Ảnh hưởng tới người khác:**

- `Object3D` thêm `track_id`.
- `VideoRecord.source` thêm `"images"`.
- `YoloeCfg` thêm `tta_flip` (mặc định tắt, khoá cache cũ giữ nguyên).
- `.env` thêm `PROJECTS_DIR`, `MM3D_PYTHON`.
- API mới: `/api/v1/projects...`, cùng mọi API review lặp lại dưới `/p/{id}/api/v1`.
- `/` giờ mở trang Dự án. Workspace nhóm vẫn ở `/ui/`.

**Cách kiểm tra:**

- `pytest` (109 passed).
- `uvicorn src.main:app` → tải lên vài ảnh jpg hoặc zip KITTI → Duyệt 2D → Xuất COCO.
- Có `.venv-mm3d` thì tải zip nuScenes một scene để có thêm bước 3D (đã chạy thử end-to-end trên CPU).

**Còn dở / việc tiếp:**

- Chạy held-out 3D trên GPU (`run3d.py run -m pointpillars ssn centerpoint_pillar centerpoint_voxel --tta`, rồi
  `eval`). Nếu `ensemble` thắng trên held-out thì giữ làm mặc định cho dự án.
- Fine-tune YOLOE trên máy 3090 theo `tools2d/README.md`, rồi chấm held-out bằng `tools2d/eval2d.py`.

## 2026-09-29 (2) · Kiên · nhánh `feat/yoloe-review-sim`

**Làm gì:** BEV cho chế độ Ảnh / Video (`V`): khung bên phải có ảnh camera chiếu xuống mặt đường (homography), điểm
LiDAR và các box 2D đặt lên mặt đường (theo LiDAR, không có thì theo chân vật). Bấm box trên BEV để chọn; sửa box trên
ảnh thì BEV cập nhật ngay. Video tải lên không có calibration: giả định camera cao 1.5 m, nhìn thẳng (UI ghi rõ).

**File chính:** `src/web/bev2d.js` (mới), `src/services/bev.py`, `src/api/routes.py`, `src/web/app.js` (`draw()` gọi
BEV, `window.AL`), `src/web/index.html`, `tests/test_api/test_bev2d.py`.

**Ảnh hưởng tới người khác:** API thêm `GET /frames/{id}/bev` và `/bev/meta`; `draw()` trong app.js tách thành
`drawCanvas()` + vẽ BEV. Không đổi schema.

**Cách kiểm tra:** `pytest` (99 passed); UI chế độ Ảnh → `V`.

**Còn dở / việc tiếp:** chưa vẽ / sửa box trực tiếp trên BEV của chế độ 2D (chỉ chọn).

## 2026-09-29 · Kiên · nhánh `feat/yoloe-review-sim`

**Làm gì:** UI 3D vẽ thêm box cho vật mô hình bỏ sót (`B`) và sửa box (`E`: kéo / phím / gõ số), tự đặt lên mặt đường
và co khít điểm LiDAR (`F`). Thêm ảnh BEV ghép 6 camera bằng homography mặt đường (`I`), trải dưới point cloud để gán
nhãn nhìn được quanh xe thay vì từng camera. Thử co khít trên một xe CenterPoint bỏ sót: tâm lệch 0.16 m so với nhãn gốc,
rộng 1.66 / 1.60 m, cao 1.76 / 1.85 m, dài 3.37 / 3.79 m (LiDAR chỉ thấy một mặt xe).

**File chính:** `src/services/bev.py` (mới), `src/services/review3d.py`, `src/models/schemas3d.py`,
`src/api/routes3d.py`, `src/web/app3d.js`, `src/web/index.html`, `tests/test_services/test_bev.py`.

**Ảnh hưởng tới người khác:** `Action3DRequest` thêm `EDIT_BOX` / `ADD_BOX` (trường `box`, `object_id` thành tuỳ
chọn); `Object3D` thêm `original_box`; log 3D thêm `source`, `final_box`; metrics 3D thêm `added`, `edited`; API thêm
`GET /3d/frames/{m}/{f}/bev`. Sửa hướng khung nhìn cho đúng hệ LiDAR nuScenes (x phải, y trước).

**Cách kiểm tra:** `pytest` (97 passed); mở UI → 🧊 3D → `I` bật ảnh BEV, `B` vẽ box, `F` co khít, `Enter` lưu.

**Còn dở / việc tiếp:** box người vẽ chưa qua QA Agent (coi là đã duyệt); chưa lan truyền box 3D sang keyframe sau.

## 2026-09-28 (7) · Kiên · nhánh `feat/yoloe-review-sim`

**Làm gì:** Thêm phần 3D vào sản phẩm. UI có chế độ 🧊 3D (khung 3D three.js + BEV + ảnh camera chiếu box, duyệt
Keep/Delete/đổi lớp, approve, metrics, export nuScenes). QA Agent 3D kiểm chứng box bằng 6 camera + LiDAR (port
`verify_objects.py` của Việt). `tools3d/run3d.py` chạy và chấm mAP/NDS 6–8 mô hình MMDetection3D (PointPillars, SSN,
CenterPoint pillar/voxel, FCOS3D, PGD, BEVFusion) trên các scene val có trên máy.

**File chính:** `src/services/{verify3d,label3d,review3d,eval3d}.py`, `src/models/schemas3d.py`, `src/api/routes3d.py`,
`src/web/app3d.js` + `src/web/vendor/` (three.js), `tools3d/`, `scripts/pack_nuscenes_subset.py` (`--split`,
`--all-cameras`, `--lidar-sweeps`).

**Ảnh hưởng tới người khác:** config thêm khối `verify3d` (có mặc định, không bắt buộc); API thêm `/api/v1/3d/*`;
CLI thêm `label3d`, `evaluate3d`; requirements thêm `scipy`. Phần 2D không đổi.

**Cách kiểm tra:** `pytest` (91 passed); `powershell -ExecutionPolicy Bypass -File tools3d\setup.ps1` rồi
`.venv-mm3d\Scripts\python tools3d\run3d.py all --dataroot ..\v1.0-trainval`; sau đó `python -m src.cli label3d
--model pointpillars` và mở UI → 🧊 3D.

**Kết quả (2026-09-29):** 6 mô hình trên 27 scene val (1076 keyframe). CenterPoint voxel tốt nhất (mAP 0.573, NDS
0.647 so với PointPillars 0.470 / 0.562) và còn nhanh hơn → đặt làm mặc định trong UI. Chi tiết trong
`eval/compare_3d.ipynb`. `label3d` dùng ngưỡng điểm riêng cho từng mô hình (`--min-score auto`).

**Còn dở / việc tiếp:** UI 3D chưa vẽ thêm box (vật mô hình bỏ sót); BEVFusion trên Windows chưa thử.

## 2026-09-28 (6) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Chạy được trên dữ liệu chưa gán nhãn: bảng nuScenes không có `sample_annotation.json` vẫn auto-label
bình thường (GT rỗng); `evaluate` báo rõ "không có GT" thay vì lỗi. Đã thử: 40 keyframe nuScenes bỏ bảng nhãn, và một
mp4 10 s không nhãn tải lên qua UI (YOLOE, 24 keyframe, 176 object, 35 bị gắn cờ).

**File chính:** `src/services/nuscenes_data.py`, `src/cli.py`, `tests/test_services/test_sequence_data.py`.

**Ảnh hưởng tới người khác:** Không.

**Cách kiểm tra:** `pytest` → 84 passed.

**Còn dở / việc tiếp:** Trên mp4 thử, lan truyền chỉ mang được ít nhãn (xe nhỏ ở xa, track dừng sớm); cần thử thêm
với video thật của nhóm.

## 2026-09-28 (5) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Đo công duyệt thật của quy trình "người duyệt frame khó, máy lan truyền phần còn lại" bằng người duyệt mô
phỏng theo GT. Phát hiện lan truyền cũ không đỡ công (ghi ra box thuần dự đoán, ~83% sai). Sửa: không ghi box lan
truyền khi detector không thấy ở keyframe đích (`propagation.emit_coasting: false`). Đặt `detection.min_score: 0.30`
làm mặc định. Cộng hai thay đổi: 23.4 → 15.1 phút cho 40 keyframe (−35%).

**File chính:** `src/services/propagation.py`, `src/models/qa_config.py`, `configs/autolabel.yaml`,
`scripts/simulate_review.py`, `eval/review_simulation.ipynb`, `eval/results/review_sim/`,
`tests/test_services/test_propagation.py`.

**Ảnh hưởng tới người khác:** Config: `min_score` 0.10 → 0.30, thêm `propagation.emit_coasting` (mặc định false).
Workspace cũ cần `python -m src.cli run --overwrite` để áp ngưỡng mới (dùng cache, không detect lại; xoá kết quả duyệt
trong workspace đó).

**Cách kiểm tra:** `pytest` → 83 passed; mở `eval/review_simulation.ipynb`.

**Còn dở / việc tiếp:** Box người vẽ thêm mà detector không thấy chưa được mang sang frame sau (cần tracker theo ảnh).

## 2026-09-28 (4) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Thêm ngưỡng giữ box sau fusion (`detection.min_score`, `min_score_per_class`) và chỉ số precision / recall
/ F1 trong `evaluate`. Notebook `eval/precision_tuning.ipynb` đo trên 2 scene nuScenes thật: nâng ngưỡng 0.1 → 0.3
tăng precision 0.34 → 0.51, F1 0.455 → 0.556, số box bị gắn cờ 260 → 88, không cần train lại.

**File chính:** `src/models/qa_config.py`, `src/services/pipeline.py`, `src/services/evaluation.py` (`evaluate_pr`),
`src/cli.py`, `configs/autolabel.yaml`, `eval/precision_tuning.ipynb`, `eval/results/thresholds/threshold_experiment.json`.

**Ảnh hưởng tới người khác:** Config có thêm `detection.min_score` (mặc định 0.10, giữ nguyên hành vi cũ) và
`min_score_per_class`. `autolabel2d_eval.json` có thêm khối `pr`. Đổi ngưỡng không làm mất cache detection.

**Cách kiểm tra:** `pytest` → 82 passed; `python -m src.cli evaluate` in thêm dòng precision / recall / F1.

**Còn dở / việc tiếp:** Nhóm quyết có đặt `min_score: 0.3` làm mặc định không (đề xuất trong notebook).

## 2026-09-28 (3) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Bỏ phần chạy trên Kaggle (nhóm chạy trên máy cá nhân): xoá `demo/kaggle_nuscenes_run.ipynb`, bỏ hướng dẫn
Kaggle trong README / docs. `scripts/pack_nuscenes_subset.py` giữ lại để chọn scene ngẫu nhiên và gửi bộ dữ liệu rút
gọn cho thành viên.

**File chính:** `demo/kaggle_nuscenes_run.ipynb` (xoá), `README.md`, `docs/demo-script.md`, `scripts/pack_nuscenes_subset.py`.

**Ảnh hưởng tới người khác:** Không (không có code nào gọi notebook Kaggle).

**Cách kiểm tra:** `pytest` → 82 passed.

**Còn dở / việc tiếp:** Không.

## 2026-09-28 (2) · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Đổi detector mặc định từ YOLO-World-L sang YOLOE-26-L theo kết quả `eval/compare_detectors.ipynb`.
YOLO-World vẫn dùng được (`--detectors yolo_world`).

**File chính:** `configs/autolabel.yaml`, `src/models/qa_config.py`, `demo/kaggle_nuscenes_run.ipynb`, README,
ARCHITECTURE, docs.

**Ảnh hưởng tới người khác:** `detection.detectors` mặc định là `[yoloe]`. Workspace đã auto-label bằng YOLO-World thì
lan truyền không tìm thấy cache detection (khoá cache theo detector): chạy lại `run` + `detect-sweeps`, hoặc đặt
`detectors: [yolo_world]` trong config khi mở workspace cũ.

**Cách kiểm tra:** `pytest` → 82 passed.

**Còn dở / việc tiếp:** Chạy YOLOE trên GPU cho nhiều scene hơn.

## 2026-09-28 · Kiên · nhánh `feat/label-propagation`

**Làm gì:** Thêm hai detector `yoloe` (YOLOE-26, open-vocab, text encoder MobileCLIP2) và `yolo26` (YOLO26, tập lớp
đóng COCO) bên cạnh `yolo_world`, cùng notebook so sánh `eval/compare_detectors.ipynb` (matplotlib) trên 2 scene
nuScenes trainval ngẫu nhiên. Kết quả: YOLOE-26-L mAP@0.5 0.45 (YOLO26 0.40, YOLO-World từ vựng COCO 0.31), là model
duy nhất nhận barrier, lan truyền 50/52 đúng với độ phủ cao nhất. Chưa đổi detector mặc định.

**File chính:** `src/services/detectors/yoloe.py`, `src/services/detectors/yolo26.py`, `src/models/qa_config.py`,
`configs/autolabel.yaml`, `src/cli.py` (`--detectors` cho `propagate`, `eval-propagation`), `scripts/bench_detectors.py`,
`eval/compare_detectors.ipynb`, `eval/results/compare/`, `requirements-ml.txt` (ultralytics>=8.4).

**Ảnh hưởng tới người khác:** Config thêm `detection.yoloe`, `detection.yolo26`. Muốn dùng YOLOE: `detectors: [yoloe]`
trong config, hoặc `python -m src.cli run --detectors yoloe`. Lần đầu tự tải `yoloe-26l-seg.pt` và `mobileclip2_b.ts`.

**Cách kiểm tra:** `pytest` → 82 passed. Mở `eval/compare_detectors.ipynb`, Run All (chỉ đọc JSON trong
`eval/results/compare/`, không cần GPU).

**Còn dở / việc tiếp:** Chạy lại YOLO-World với đủ prompt trên GPU (ô "Chạy lại" trong notebook); thử ensemble
`[yoloe, yolo26]`.

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
