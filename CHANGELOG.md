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

## 2026-10-02 · Kiên · nhánh `kien-mentor`

**Làm gì:** Theo góp ý mentor: (1) làm việc nhiều người kiểu CVAT / Google Docs — tài khoản + đăng nhập, mời vào dự án
bằng link (chỗ cắm SMTP), vai trò owner / reviewer / annotator, chia frame cho từng người, khoá frame đang mở (người khác
chỉ xem, có "lấy quyền sửa"), hàng đợi lọc "Của tôi"; (2) thẻ object 2D / 3D gọn kiểu CVAT (một dòng #id · lớp · rủi ro ·
nút, lỗi QA thành chip rê chuột xem); (3) box 3D bị che ≥ 50% hoặc cắt khỏi ảnh ≥ 60% vẽ nét đứt trên camera; (4) BoT-SORT
cho lan truyền 2D (`propagation.association: botsort`: ngoại hình + bù chuyển động camera); (5) DAM4SAM (SAM 2.1) làm
nguồn dự đoán box cho tracker (`propagation.flow: dam4sam`, cần GPU + repo DAM4SAM); (6) TrackEval: HOTA / MOTA / IDF1 cho
nhãn 2D theo track_id và box 3D, bộ đo trong repo khớp TrackEval chính thức, tab Metrics có bảng tracking. Kèm các sửa
trước đó: phát video mượt 10 fps (GET /videos/{id}/playback), dải sweep t-2…t+2 hiện lại, bấm box trên ảnh camera → box 3D.

**File chính:** `src/services/users.py`, `src/api/auth_routes.py`, `src/api/auth_middleware.py` (mới); `src/services/botsort.py`,
`src/services/trackeval.py`, `src/services/dam4sam.py`, `tools2d/dam4sam.py` (mới); `src/services/propagation.py`,
`src/services/sequence.py`, `src/services/verify3d.py`, `src/web/{app,app3d,projects}.js`, `src/web/login.html`.

**Ảnh hưởng tới người khác:** `Verify3D` thêm `visible_by_cam`, `occlusion_by_cam` (mặc định rỗng); `Project` thêm `members`;
`PropagationCfg` thêm `association: botsort`, `flow: dam4sam`, `botsort_*`, `dam4sam_model`; `.env` thêm `AUTH_REQUIRED`,
`AUTH_OPEN_SIGNUP`, `USERS_FILE`, `PUBLIC_URL`, `SMTP_*` (mặc định tắt: chạy một người như cũ). API mới: `/api/v1/auth/*`,
`/api/v1/projects/{id}/members|invites|assignments|split`, `/api/v1/invites/{token}`, `/frames/{id}/lock`, `/presence`,
`/metrics/tracking`, `/videos/{id}/playback`. Khi đã đăng nhập, POST sửa frame mà người khác đang giữ khoá → 409 FRAME_LOCKED.

**Cách kiểm tra:** `pytest` (211 pass), `ruff check src tests`; `python -m src.cli trackeval --export-mot`;
`scripts\tasks.ps1 evaltemporal` (có thêm cấu hình always+botsort, off+botsort); `scripts\tasks.ps1 dam4sam` (GPU);
UI: đăng ký ở /ui/login.html → trang Dự án → Thành viên → tạo link mời → mở link ở trình duyệt khác → Chia việc.

**Còn dở / việc tiếp:** chạy BoT-SORT / DAM4SAM / TrackEval trên tập test (GPU) để có số; ngưỡng ngoại hình BoT-SORT
(`botsort_appearance`) mới chọn tay, cần dò trên dev; gửi mail cần SMTP; chưa có đổi mật khẩu / quên mật khẩu.

## 2026-10-01 (2) · Kiên · nhánh `kien`

**Làm gì:**

- **Bảng so sánh mọi phương pháp đã dùng** (2D, 3D; chỉ số quyết định, có / không dùng thì tăng / giảm bao nhiêu):
  `eval/results/bang-so-sanh.md`. Đo thêm trên test 20 scene: lan truyền 2D không tracker / ByteTrack / OC-SORT /
  ByteTrack + OC-SORT, có và không optical flow (`eval/results/tracker_test20.json`). ByteTrack vẫn tốt nhất (điểm 2180).
- Kết quả `eval2d` của detector fine-tune lưu vào `eval/results/det2d_finetune.json`.
- PRD: lịch sử tối ưu thêm dòng 17–18, bảng detector 2D, kết quả đo.
- **Gate G2:** README thêm bảng biến môi trường và ví dụ gọi API kèm output thật; `.env.example` chỉ giữ biến dự án
  dùng; `docs/eval-evidence.md` gồm 11 test case thủ công với output thực tế (1 lỗi đã biết: kiểm tra temporal với vật ở
  gần chạy nhanh qua ảnh); sơ đồ kiến trúc cập nhật + `docs/architecture.png`.

**File chính:** `eval/results/bang-so-sanh.md`, `eval/results/det2d_finetune.json`, `eval/results/tracker_test20.json`,
`PRD-AutoLabel3D.md`, `README.md`, `.env.example`, `docs/eval-evidence.md`, `docs/architecture_diagram.md`.

**Ảnh hưởng tới người khác:** `.env.example` bỏ các biến template không dùng (OpenAI, database, Chroma); còn lại chỉ tài liệu và số liệu.

**Cách kiểm tra:** mở `eval/results/bang-so-sanh.md`.

**Còn dở / việc tiếp:** chạy lại lan truyền 2D và QA với detector mới (`scripts\tasks.ps1 evaltemporal`).

## 2026-10-01 · Kiên · nhánh `kien`

**Làm gì:** Detector 2D mặc định đổi sang YOLOE-26L fine-tune trên nuImages (linear probe, 1280 px, 10 epoch trên RTX
3090). Chấm trên nuScenes CAM_FRONT (`tools2d/eval2d.py`, ngưỡng 0.30, IoU 0.5):

| | dev (119 ảnh) | held-out (957 ảnh) | P held-out | R held-out |
|---|---|---|---|---|
| zero-shot `yoloe-26l-seg.pt` | 0.392 | 0.312 | 0.499 | 0.523 |
| linear probe nuImages | **0.429** | **0.366** | 0.484 | **0.581** |

AP50 held-out tăng mạnh ở lớp yếu: barrier 0.13→0.33, bicycle 0.12→0.23, traffic_cone 0.39→0.50, motorcycle
0.21→0.28, pedestrian 0.23→0.29; giảm nhẹ truck 0.52→0.47, bus 0.81→0.78; construction_vehicle / trailer vẫn ~0.

**File chính:** `configs/autolabel.yaml` (`detection.yoloe.weights`), `weights/yoloe-26l-nuimages-lp-1280.pt` (commit
trước, kèm `.sha256` và `results-lp-1280.csv`), `tools2d/train_pc.ps1` (sửa hàm `Data` trùng từ khoá PowerShell).

**Ảnh hưởng tới người khác:** Config đổi detector mặc định. Trọng số mới là **tập lớp đóng 10 lớp nuScenes**: prompt chữ /
lớp thêm mới không có tác dụng; cần open-vocab thì đặt lại `yoloe-26l-seg.pt`. Cần `git pull` để có file `.pt`
(50.6 MB, nằm ngoài `.gitignore` nhờ `git add -f`). Cache detection cũ không bị dùng lại (khoá theo tên trọng số).

**Cách kiểm tra:** `powershell -ExecutionPolicy Bypass -File scripts\tasks.ps1 eval2d -Weights weights\yoloe-26l-nuimages-lp-1280.pt`

**Còn dở / việc tiếp:** Bản fine-tune toàn mạng (`full-1280`) đang train trên PC 3090; xong thì chấm lại cùng lệnh, giữ
bản nào held-out cao hơn.

## 2026-09-30 (2) · Kiên · nhánh `kien`

**Làm gì:**

- **Tab ⚙ Cài đặt** trên UI (mở cả từ thẻ dự án): chỉnh optical flow, cách ghép của tracker, tính lại score theo sweep,
  ngưỡng giữ box, số keyframe lan truyền; lưu theo workspace / dự án. Nút **Áp dụng lại** và **Chạy đánh giá** (trước /
  sau) chạy nền — không phải sửa config hay mở terminal.
- **Tracker lan truyền 2D ghép hai lượt kiểu ByteTrack** (mặc định). Dev: box sai 224 → 203; held-out 4 scene: 329 → 300.
- **Optical flow đo lại trên 20 scene held-out (795 keyframe, GPU):** nhãn lan truyền đúng 2910 → 3329 (+14%), đổi ID
  265 → 97. Thử "tin box sweep score thấp" cho QA: lỗi lọt qua tăng, không bật.
- **Lan truyền box 3D đã duyệt:** approve một keyframe 3D thì box sang các keyframe sau (bù chuyển động xe + vận tốc,
  lớp / kích thước theo người, box người xoá tự xoá). UI 3D: ô "↦ Lan truyền", phím `T`. Held-out 4 scene: ~95% box
  đúng vật, đổi ID 98 → 36.
- **Đo thời gian từng bước** (2D, 3D, nạp model, làm mờ ảnh): bảng ở tab Metrics, lệnh `profile`. CPU: detect
  17.7 s/frame; GPU laptop: 2D ~1 s, 3D ~4 s/frame.
- `tasks.ps1 serve -Workspace` để demo; ẩn cảnh báo `'half' is deprecated`.
- **Ảnh BEV dễ nhìn hơn** (3D và 2D): ghép mặt đường từ ±4 keyframe cùng scene theo ego pose, mỗi ô lấy từ lần camera
  nhìn gần nhất — lấp vùng bị xe che, hết nhoè ở xa, 2D đặt vạch đường đúng chỗ hơn; bỏ vệt cốp / capô xe mình, dọn mảnh vụn.
- **Mô hình 3D trên tập test** (24 scene, 957 keyframe, so với nhãn gốc): ensemble 4 LiDAR + track mAP 0.668 / NDS 0.713,
  CenterPoint voxel 0.578 (`eval/results/det3d_heldout.md`).
- **Thử ý tưởng VESPA** (box 3D từ box 2D + LiDAR, hướng theo chuyển động, cỡ theo lớp): không tăng mAP, không bật
  (`eval/results/vespa.md`; `src/services/fill3d.py` giữ để thử tiếp).
- **Thử OC-SORT** (OCR / ORU / OCM) cho tracker lan truyền 2D và 3D, đo trước / sau trên dev và test: 2D không lợi (tắt,
  bật được ở tab Cài đặt); 3D bật phần giữ track qua che khuất `propagation3d.max_misses: 2 → 4` — test 24 scene nhãn đúng
  +1.6%, đổi ID 470 → 543 (`eval/results/ocsort.md`).
- PRD: trạng thái từng FR, mô hình đã thử, lịch sử 16 lần tối ưu, kết quả đo, giải thích công cụ; slide MVP + so sánh.

**File chính:**

- Mới: `src/services/ui_settings.py`, `jobs.py`, `relabel.py`, `temporal_eval.py`, `timing.py`, `profiling.py`,
  `propagation3d.py`, `propagation3d_eval.py`, `fill3d.py`, `tools3d/eval_propagation3d.py`,
  `eval/results/propagation3d.md`, `det3d_heldout.md`, `vespa.md`, `ocsort.md`, `tests/test_services/test_ocsort.py`.
- Sửa: `propagation.py`, `pipeline.py`, `bev.py`, `routes.py`, `routes3d.py`, `projects.py`, `label3d.py`, `privacy.py`,
  `cli.py`, `web/app.js`, `web/app3d.js`, `configs/autolabel.yaml`, `scripts/tasks.ps1`, `PRD-AutoLabel3D.md`.

**Ảnh hưởng tới người khác:**

- Schema: `FrameRecord.autolabel_timing`, `Frame3DRecord.autolabel_timing` / `propagated_from` / `propagated_at` /
  `prelabel`; `Object3D.source` thêm `"propagated"`, `Object3D.propagation`.
- Config: `propagation.association`, `byte_high_score`, `byte_low_iou`, `oc_*`; `qa.temporal.sweep_min_score`;
  `propagation3d` (`max_misses` mặc định 4, `oc_*`).
- `routes.get_config(request)` áp thêm `<workspace>/settings.json`; có thêm `get_base_config`. Bước gán nhãn của dự án
  cũng dùng cài đặt này.
- API mới: `/settings`, `/relabel`, `/eval-temporal`, `/timing`, `POST /3d/frames/{m}/{f}/propagate`.
- `bev.bev_mosaic(..., neighbors=)`, `bev.bev_camera(..., neighbors=)`; `GET /3d/frames/{m}/{f}/bev?fuse=false` để lấy ảnh một frame như cũ.

**Cách kiểm tra:**

- `pytest` (152 pass), `ruff check src tests`.
- UI: tab ⚙ Cài đặt → đổi một mục → Lưu → Áp dụng lại → Chạy đánh giá; chế độ 3D: approve có ô "↦ Lan truyền".
- `scripts\tasks.ps1 profile`, `scripts\tasks.ps1 evalprop3d` (đủ 27 scene val).

**Còn dở / việc tiếp:** Chạy `evalprop3d` trên 27 scene; nhóm quyết định có bật `rescore: mean` không.

---

## 2026-09-30 · Kiên · nhánh `kien`

**Làm gì:** Đưa ý tưởng *Deep Feature Flow* (arXiv:1611.07715) vào sản phẩm ở mức box (optical flow OpenCV DIS, không
train lại detector) và cho sửa nhãn ở các sweep t−2…t+2:

- **Lan truyền nhãn dùng optical flow** (`propagation.flow: always`, mặc định bật). Held-out: +88 nhãn lan truyền
  đúng (+11%), đổi ID 26 → 11, mất dấu −23%.
- **QA temporal dùng flow + tính lại score theo sweep** (`qa.temporal.flow`, `qa.temporal.rescore`): có sẵn, **tắt
  mặc định**. mAP không đổi; bật `flow + mean` bớt ~60% box phải xem tay nhưng lỗi còn lại sau duyệt +5–7%.
- **Sửa tự do ở sweep** trên UI 2D: click ô t±1/t±2, keep / xoá / đổi lớp / sửa box / vẽ thêm / khôi phục, có
  hoàn tác. Keyframe được tính lại ngay (FLICKER, RECOVERED_BY_TRACK, risk). Tracker lan truyền dùng bản đã sửa.
  Box ở sweep không được xuất.
- Bảng trước / sau (dev 119 + held-out 159 keyframe): `eval/results/temporal/report.md`.

**File chính:**

- Mới: `src/services/flow.py`, `src/services/temporal_fusion.py`, `src/services/sweep_review.py`,
  `tools2d/eval_temporal.py`.
- Sửa: `pipeline.py`, `agents/nodes/temporal.py`, `propagation.py`, `sequence.py`, `routes.py`, `web/app.js`.

**Ảnh hưởng tới người khác:**

- Schema:
  - `Detection.det_score`, `LabelObject.det_score`;
  - `SweepInfo.boxes` (list `SweepBox`, None = chưa sửa);
  - `SweepActionRequest`.
- API: `POST /frames/{id}/sweeps/{offset}/actions`.
- Config: `qa.temporal.flow`, `flow_scale`, `rescore`; `propagation.flow`, `flow_scale`.
- `evaluate_propagation(..., start_every=)`.
- `WorkspaceSequenceSource(..., dataroot)`.
- Báo cáo năng suất có cột "Box sweep đã sửa". Frame cũ vẫn đọc được.

**Cách kiểm tra:**

- `pytest` (129 pass), `ruff check src tests`.
- UI: mở một frame, click ô t−1 ở dải dưới, xoá / vẽ box, xem keyframe đổi cờ.
- `python tools2d/eval_temporal.py --dataroot <nuscenes> --workspace <ws đã run> --out <thư mục>`.

**Còn dở / việc tiếp:**

- Nhóm quyết định có bật `flow + mean` (đổi chất lượng lấy công duyệt) hay không.
- Đo lại trên GPU với nhiều scene hơn.

---

## 2026-09-29 (2) · Danh · nhánh `danh`

**Làm gì:** Tái cấu trúc và hiện đại hóa toàn diện giao diện web (`src/web/`) theo hướng tinh gọn, loại bỏ hoàn toàn giao diện dư thừa. Chuẩn hóa toàn bộ tên class/ID/biến trong code. Hoàn thiện đầy đủ chức năng tương tác cho nút bấm và modal trên trang: modal thêm nhãn mới với bảng chọn màu, chỉnh sửa/xóa nhãn trực tiếp, chuyển đổi chế độ xem Dạng thô (Raw JSON) và Trình dựng nhãn (Constructor), dropdown menu thao tác dự án, bộ lọc trạng thái nhanh, bộ lọc theo lớp nhãn, sắp xếp danh sách và tạo task mới.

**File chính:**
- Sửa: `src/web/frames.html` (chuẩn hóa taxonomy xe tự hành, bỏ nút khung xương, gắn kết nối modal thêm nhãn `modal-overlay`/`modal-card`, dropdown thao tác, sắp xếp, lọc nhanh, lọc nhãn, tạo task), `src/web/ui-flow.js` (logic taxonomy, xử lý modal toàn cục `window.openModal`/`closeModal`, tìm kiếm, sắp xếp rủi ro QA / ID / đối tượng, lọc trạng thái, lọc nhãn, duyệt nhanh frame qua API `approve`), `src/web/projects.html` (dọn dẹp thanh header, bổ sung dropdown sắp xếp và lọc dự án), `src/web/export.html` & `src/web/index.html` (dọn dẹp các nút dư thừa trên topbar), `src/web/ui-flow.css` & `src/web/styles.css` (chuẩn hóa token CSS và class name không dùng `cvat-`), `src/main.py` (chuyển hướng `/flow` về `/ui/projects.html`).

**Ảnh hưởng tới người khác:** Không. Giữ nguyên toàn bộ schema Pydantic, các cấu hình trong `configs/autolabel.yaml` và các endpoint API `/api/v1/*`.

**Cách kiểm tra:**
- `ruff check src tests` → All checks passed.
- `python -m pytest tests/test_api/` → 11 passed.
- Mở server `python -m src.demo --reset` hoặc `uvicorn src.main:app`, truy cập `http://localhost:8000/ui/`:
  - Kiểm tra thanh header trên tất cả các trang gọn gàng, chỉ chứa các liên kết thực tế (Dự án, Tasks & Frames, Xuất dữ liệu), không còn nút GitHub và nút trợ giúp thừa.
  - Tại `/ui/frames.html`: Bấm **`+ Thêm nhãn`** → Modal xuất hiện ở giữa màn hình cho phép nhập tên và chọn màu; danh mục nhãn hiển thị đúng taxonomy xe tự hành, có thể sửa tên hoặc xóa nhãn; các dropdown **Thao tác**, **Sắp xếp**, **Lọc nhanh**, **Lọc theo nhãn** hoạt động phản hồi chính xác.

**Còn dở / việc tiếp:**
- Tiếp tục tối ưu hóa hiệu năng render canvas khi tải lượng lớn frame video liên tục.

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
## 2026-09-29 (4) · Kiên · nhánh `kien`

**Làm gì:** Làm các yêu cầu còn thiếu của PRD không phụ thuộc giao diện:

- **FR-15:** reject kèm lý do bắt buộc (2D + 3D).
- **FR-09:** undo / redo (Ctrl+Z / Ctrl+Y).
- **FR-19:** báo cáo CSV.
- **FR-27:** năng suất theo người / phiên, và throughput auto-label theo phiên chạy.
- **FR-17:** xuất KITTI. Đã kiểm tra đọc lại bằng `KittiDB`: tâm box lệch ≤ 5 cm (làm tròn 2 chữ số theo chuẩn KITTI),
  hướng ≤ 0.002 rad.
- **FR-04:** mask sơ bộ từ YOLOE-seg, tô trên UI và xuất COCO `segmentation`.
- **FR-03:** làm mờ mặt / biển số trước khi gửi ảnh.
- **FR-06:** thanh trượt ngưỡng score.

Ngoài ra:

- Trang Dự án kiểm tra trước thư viện còn thiếu và báo rõ Python nào đang chạy server.
- Thêm `scripts/tasks.ps1` gom các việc chạy trên terminal.

**File chính:**

- mới: `src/services/history.py`, `productivity.py`, `export_kitti.py`, `privacy.py`, `scripts/tasks.ps1`;
- sửa: `src/api/routes.py`, `routes3d.py`, `projects_routes.py`, `src/services/review.py`, `store.py`,
  `pipeline.py`, `video.py`, `label3d.py`, `detectors/{__init__,yoloe,fusion}.py`, `exporter.py`,
  `src/web/{index.html,app.js,app3d.js,projects.js,styles.css}`.

**Ảnh hưởng tới người khác:**

- `FrameRecord` / `Frame3DRecord`:
  - `status` thêm `"rejected"`;
  - thêm `reject_reason`, `rejected_by`, `rejected_at`, `autolabel_s`, `autolabel_run`.
- `Detection` / `LabelObject` thêm `mask`.
- Config thêm `privacy` và `detection.yoloe.masks`. Khoá cache detection giữ nguyên.
- `requirements-ml.txt` thêm `open-image-models`.
- `/api/v1/metrics` thêm `productivity`; `frames` thêm `rejected`.
- API mới: `reject`, `undo`, `redo`, `history`, `report.csv`, `3d/export-kitti`, `3d/exports/{file}`,
  `projects/env`, `projects/{id}/export?format=kitti`.
- Workspace thêm `events.jsonl`, `history/`, `anon/`.

**Cách kiểm tra:**

- `pytest` (121 passed).
- `scripts\tasks.ps1 check`.
- UI 2D: `K` / `D`, Ctrl+Z, nút **Trả lại**.
- Tab Metrics: bảng năng suất và nút CSV.
- 3D: **Xuất KITTI**.

**Còn dở / việc tiếp:**

- FR-21 đăng nhập / phân vai: chờ chốt giao diện với Danh.
- FR-07 / FR-08: nối box 2D ↔ 3D.
- FR-10 brush sửa mask, FR-25 nội suy 3D, FR-20 DVC.

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
