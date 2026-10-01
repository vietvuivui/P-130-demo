# AutoLabel 3D — Auto-label nuScenes 2D + 3D, QA Agent đa tín hiệu, Review by exception

> Gán nhãn 2D cho xe tự lái tốn thời gian vì người phải soát từng box → model open-vocab tự sinh box,
> QA Agent kiểm chứng chéo bằng LiDAR và các frame camera lân cận để chấm rủi ro, người chỉ tập trung
> vào số ít box rủi ro cao và duyệt theo lô phần còn lại.

![Workflow](docs/workflow.png)

## Workflow

| Bước | Làm gì | Code |
|---|---|---|
| 1. Input | CAM_FRONT keyframe, sweep t-2..t+2 (12Hz), LIDAR_TOP, calibration + ego pose từ nuScenes v1.0-mini | `src/services/nuscenes_data.py` |
| 2. 2D Detection | YOLOE-26 (mặc định), YOLO-World, Grounding DINO, Florence-2 — open-vocab, không train; YOLO26 (tập lớp COCO) làm model phụ; nhiều model thì fuse ([so sánh](eval/compare_detectors.ipynb)) | `src/services/detectors/` |
| 3. QA Agent | LangGraph: 3.1 confidence · 3.2 LiDAR support · 3.3 temporal (song song) → 3.4 issue → 3.5 risk | `src/agents/` |
| 4. Review by exception | Low risk duyệt theo lô, high risk xem chi tiết: Keep / Delete / Change class / Sửa box / Add box | `src/web/`, `src/services/review.py` |
| 5. Correction log | Mỗi thao tác ghi 1 dòng JSONL: dự đoán, risk, issue, hành động, kết quả cuối | `data/workspace/corrections.jsonl` |
| 6. Dataset | Chỉ frame đã approve và sạch QC: COCO + JSONL + log + báo cáo QA + manifest (READY, SHA256) | `src/services/exporter.py` |
| 7. Lan truyền video (chế độ 🎞 Video) | Approve một keyframe → nhãn của người được mang sang các keyframe sau (track qua mọi ảnh 12Hz / 10 fps), dừng trước frame người đã mở. Nguồn: scene nuScenes hoặc mp4 tải lên | `src/services/propagation.py`, `src/services/sequence.py`, `src/services/video.py` |
| 8. QC | Kiểm lại nhãn cuối sau khi người sửa (chặn approve khi còn lỗi), audit ngẫu nhiên phần duyệt theo lô, checklist READY trước khi xuất, Quick Check file nhãn từ ngoài | `src/services/qc/` |

**Issue code:** `LOW_CONFIDENCE`, `CLASS_CONFLICT`, `NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH`, `FLICKER`,
`RECOVERED_BY_TRACK`, `BOX_TOO_LARGE`, `ASPECT_RATIO_ABNORMAL`; nhãn lan truyền thêm `PROP_LOW_CONF`,
`PROP_COASTING`, `PROP_CLASS_DIFFERS`.

**Risk:** `risk = w1(1 − score) + w2·lidar + w3·temporal + w4·geometric` → low < 0.30 ≤ medium < 0.60 ≤ high.
Object có issue luôn ≥ 0.30 nên không bao giờ bị duyệt theo lô. Mọi ngưỡng và trọng số nằm trong
[configs/autolabel.yaml](configs/autolabel.yaml). Chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md).

## Demo nhanh (không cần GPU, không cần nuScenes)

```bash
pip install -r requirements.txt
python -m src.demo            # hoặc: make demo
# mở http://localhost:8000 → chọn 🎞 Video trên thanh trên cùng
```

Lệnh này sinh một video đường phố tổng hợp 16 giây (`data/demo/street_demo.mp4`), cho nó đi qua đúng đường xử lý
của video tải lên (cắt frame 10 fps, keyframe 2 fps, detect, QA Agent) bằng một detector demo tìm vật thể theo
màu, rồi mở UI trên workspace riêng `data/demo/workspace` (không đụng dữ liệu thật). Kịch bản trình diễn từng
bước: [docs/demo-script.md](docs/demo-script.md). `python -m src.demo --reset` để sinh lại từ đầu.

## Web cho end-user: Dự án (tải dữ liệu lên → gán nhãn → duyệt → xuất)

```bash
uvicorn src.main:app --port 8000      # mở http://localhost:8000 → trang Dự án
```

**Làm việc nhiều người** (kiểu CVAT / Google Docs, `src/services/users.py`): đăng ký ở `/ui/login.html` (người đầu tiên là
admin); ở thẻ dự án bấm **Thành viên** → nhập email, chọn vai trò → **Tạo link mời** (gửi mail nếu có `SMTP_*`, không thì
copy link gửi tay); người được mời mở link, tạo tài khoản là vào dự án; **Chia việc** chia đều frame chưa duyệt cho các
thành viên đã tick. Trong trang gán nhãn: chip **Của tôi** lọc frame được giao, tên người duyệt lấy theo tài khoản, ai đang
mở frame nào hiện 🔒 ở hàng đợi; frame người khác đang mở chỉ xem được (có "Lấy quyền sửa"). Vai trò: owner mời / chia việc,
reviewer duyệt / trả lại / chia việc, annotator gán nhãn. Bật `AUTH_REQUIRED=1` để bắt buộc đăng nhập.

Mỗi dự án có một thư mục riêng `data/projects/<id>/`, gồm:

- file tải lên;
- dữ liệu đã đổi sang nuScenes;
- workspace nhãn;
- log.

Các bước chạy nền, lần lượt, trên một GPU. Trang Dự án hiện tiến độ từng bước. Bấm **Duyệt 2D** / **Duyệt 3D** để mở
UI review quen thuộc trên đúng dự án đó (`/ui/?project=<id>`), rồi **Xuất nuScenes** / **Xuất COCO** để tải về file zip.

| Dữ liệu tải lên | Nhận dạng | Các bước |
|---|---|---|
| Video mp4 / mov / avi / mkv / webm | tự động | cắt 10 fps → gán nhãn 2D keyframe 2 fps + QA → lan truyền khi duyệt |
| Ảnh jpg / png (nhiều file hoặc .zip) | tự động | gán nhãn 2D từng ảnh (hoặc coi là frame liên tiếp của video) |
| nuScenes (.zip có `v1.0-*/`, `samples/`, `sweeps/`) | `v1.0-*/scene.json` | 2D (CAM_FRONT) + 3D |
| KITTI object (`velodyne/ image_2/ calib/`) hoặc tracking (`velodyne/0000/ image_02/0000/ calib/0000.txt`, `oxts/`) | cấu trúc thư mục | đổi sang nuScenes (trục, calib, pose từ oxts) → 2D + 3D |
| LiDAR rời + camera (định dạng dưới) | `calib.json` + `lidar/` | đổi sang nuScenes → 2D + 3D |

Với dữ liệu có LiDAR, bước 3D có hai phần:

- **Dự đoán:** gộp 4 mô hình LiDAR rồi tinh chỉnh theo track, chạy bằng `tools3d/run3d.py predict` trong môi trường
  `.venv-mm3d`. Máy chưa cài môi trường này thì dự án chỉ có nhãn 2D và trang báo rõ lý do.
- **Kiểm chứng:** kiểm tra từng box bằng camera để xếp mức rủi ro.

Xuất nuScenes cho ra một zip gồm:

- `sample_annotation.json` + `instance.json`: nhãn chuẩn nuScenes, box cùng `track_id` được nối thành một instance;
- các bảng gốc, để mở được bằng nuscenes-devkit;
- `labels3d_nusc.json`: định dạng kết quả detection;
- log sửa của người duyệt.

**Định dạng "LiDAR rời + camera"** (zip; tên frame giống nhau giữa `lidar/` và các thư mục camera):

```
calib.json
lidar/000000.bin | .pcd | .npy     float32 x y z [cường độ]
CAM_FRONT/000000.jpg               mỗi camera một thư mục, tên = tên camera trong calib.json
poses.json       (tuỳ chọn)        {"000000": [[4x4 ego -> thế giới]], ...}   có thì gộp được nhiều lần quét + track
timestamps.json  (tuỳ chọn)        {"000000": 12.30, ...} giây; không có thì dùng frame_rate
```

```json
{
  "frame_rate": 10,
  "lidar": {"axes": "x_forward", "to_ego": [[1,0,0,0],[0,1,0,0],[0,0,1,1.8],[0,0,0,1]]},
  "cameras": {
    "CAM_FRONT": {"intrinsic": [[1266,0,816],[0,1266,491],[0,0,1]],
                  "lidar_to_camera": [[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]}
  }
}
```

- `axes` là `x_forward` (x trước, y trái, như KITTI / Velodyne) hoặc `x_right` (như nuScenes).
- `to_ego` mặc định là LiDAR cao 1.8 m, cùng hướng xe.
- Camera đầu tiên dùng cho gán nhãn 2D.

API: `GET/POST /api/v1/projects`, `GET/DELETE /api/v1/projects/{id}`, `POST /api/v1/projects/{id}/run`,
`POST /api/v1/projects/{id}/export?format=nuscenes|coco`, `GET /api/v1/projects/{id}/exports/{file}`. Mọi API review
ở bảng dưới cũng dùng được cho từng dự án với tiền tố `/p/{id}/api/v1/...`.

## Quick Start (nuScenes thật)

Yêu cầu: Python 3.11, GPU NVIDIA (chạy được trên RTX 3060 6GB; CPU cũng chạy nhưng chậm),
nuScenes v1.0-mini giải nén vào `./v1.0-mini-001` (hoặc đặt `NUSCENES_DATAROOT` trong `.env`).

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-ml.txt
python scripts/download_weights.py   # YOLOE-26-L + text encoder MobileCLIP2 -> weights/ (kiểm SHA256)
pip install git+https://github.com/ultralytics/CLIP.git  # tokenizer cho YOLOE / YOLO-World (lần đầu tự tải yoloe-26l-seg.pt + mobileclip2_b.ts)

# 1-3. Auto-label + QA Agent (detection được cache theo model, chạy lại rất nhanh)
python -m src.cli run --limit 40                 # 40 keyframe đầu
python -m src.cli run                            # cả 404 keyframe
python -m src.cli run --detectors yolo_world --overwrite   # đổi model (detection cũ vẫn trong cache)
python -m src.cli run --detectors yoloe yolo26 --overwrite   # ensemble (YOLO-World: --detectors yolo_world)

# Đánh giá so với GT (mAP + flag recall/precision) -> eval/results/autolabel2d_eval.md
python -m src.cli evaluate

# 4-6. UI review
uvicorn src.main:app --port 8000
# mở http://localhost:8000

# 7. Lan truyền nhãn trên video (tuỳ chọn thêm detect-sweeps để tracker có detection ở mọi ảnh 12Hz)
python -m src.cli detect-sweeps --scenes scene-0061        # GPU; `run` đã cache sẵn sweep t-2..t+2
python -m src.cli propagate scene-0061_000                 # hoặc bấm "Lan truyền" / tự chạy khi approve trong UI
python -m src.cli eval-propagation                         # -> eval/results/propagation_eval.md
```

Chỉ chạy UI / test (không cần GPU): `pip install -r requirements.txt`, rồi `pytest` hoặc `uvicorn`.

Test nhanh trên vài scene ngẫu nhiên của bản trainval (chỉ chọn scene có ảnh trên máy), rồi auto-label đúng các scene đó:

```bash
python scripts/pack_nuscenes_subset.py --dataroot ../v1.0-trainval --version v1.0-trainval \
    --random 2 --seed 20260927 --window 20 --with-lidar --workspace none --out ../nusc_random.zip
```

Giải nén zip vào thư mục repo, trỏ `.env` vào `autolabel_subset/` (README.txt trong zip), chạy `run` như trên. Bản rút
gọn (~60 MB) cũng dùng để gửi cho thành viên chưa tải đủ 48 GB dataset.

## UI

Thanh trên cùng có ba chế độ:

- **🖼 Ảnh** — duyệt từng frame như MVP: hàng đợi sắp theo risk, review by exception. Không lan truyền.
- **🎞 Video** — danh sách video (mỗi scene nuScenes là một video; mp4 tải lên bằng nút *⬆ Tải lên mp4*),
  timeline riêng cho video đang mở, lan truyền nhãn giữa các frame.
- **🧊 3D** — duyệt box 3D do mô hình LiDAR / camera sinh ra (chọn mô hình ở góc trên), xem mục *Phần 3D* bên dưới.

Chế độ Ảnh:

- **Hàng đợi frame** sắp theo frame risk (khó nhất lên đầu), lọc theo trạng thái.
- **Canvas**: box tô màu theo risk (xanh/vàng/đỏ, luôn có nhãn chữ), box `RECOVERED_BY_TRACK` nét đứt,
  bật overlay điểm LiDAR (màu theo độ sâu) và GT để đối chiếu.
  Box GT (nét đứt trắng, chữ `GT <lớp>`) = box 3D của nuScenes chiếu xuống ảnh (hình chữ nhật bao 8 đỉnh) nên thường
  rộng hơn box detector; box GT mờ không chữ = vật hiển thị 0–40% hoặc không có điểm LiDAR/radar, bỏ qua khi đánh giá.
- **Zoom**: lăn chuột trên ảnh (phóng quanh con trỏ), `+`/`−`, `0` về vừa khung, hoặc nút `− 100% +`; khi phóng to
  kéo vùng trống để di chuyển ảnh. Vẽ/sửa box vẫn đúng toạ độ ở mọi mức zoom.
- **Temporal strip** t-2 … t+2: xem object đang chọn có/không ở từng sweep; click để mở sweep đó và **sửa tự do**
  (keep / xoá / đổi lớp / sửa box / vẽ thêm, cùng phím tắt, `Esc` quay về keyframe). Box ở sweep không được xuất;
  sửa để FLICKER / RECOVERED_BY_TRACK, risk và lan truyền của keyframe đúng hơn — keyframe được tính lại ngay.
- **Panel risk**: High (card chi tiết + crop + issue + giải thích), Medium, Low (thu gọn + *Approve all low-risk*).
- **Correction log** và **Metrics & Export**: M4 (tỉ lệ nhãn phải sửa), M1 (thời gian/frame),
  flag precision/recall của agent tính từ thao tác thật của người, precision của từng issue code.

Chế độ Video:

- **Timeline**: mỗi keyframe một thẻ (ảnh thu nhỏ, thời điểm, số object chờ duyệt, vạch màu risk). ★ = người đã
  duyệt, ↦ = nhãn lan truyền chưa ai mở, ✎ = đang sửa. Bấm vào thẻ, hoặc **kéo thẻ thả lên khung ảnh**, để mở
  frame đó và chỉnh y như chế độ Ảnh (sửa box, đổi lớp, xoá, vẽ thêm). `◀ ▶|` từng frame, *▶ Phát* xem nhanh.
- **Lan truyền**: bật "Lan truyền khi approve" (mặc định bật) thì approve xong một frame, nhãn được mang sang
  các frame sau của video và UI mở luôn frame kế tiếp chưa duyệt. Nhãn lan truyền ghi `↦ c 0.82`
  (c_prop = độ tin cậy lan truyền); box nét chấm là box dự đoán khi detector không thấy. Box người đã xoá được tự
  xoá ở frame sau ("✗ Tự xoá theo frame #1", bấm Keep để khôi phục). Nút *↦ Lan truyền* (`T`) chạy lại bằng tay.
- **Tải lên mp4**: server cắt frame ngay, auto-label + QA chạy nền; danh sách hiện tiến độ và báo khi xong.
  Video không có LiDAR/calibration nên các check LiDAR tự bỏ qua, risk dựa trên score, temporal và hình học.

**BEV** (`V`, cả chế độ Ảnh và Video): khung bên phải nhìn từ trên xuống. Ảnh camera được chiếu xuống mặt đường bằng
homography (nuScenes: ngoại tham số thật của camera, mặt đường theo LiDAR, ghép thêm các keyframe lân cận như ảnh BEV
3D; video tải lên: giả định camera cao 1.5 m, nhìn thẳng), trên đó là
điểm LiDAR và từng box 2D đặt lên mặt đường: có điểm LiDAR trong box thì theo độ sâu LiDAR (nét liền), không thì theo
chân vật chạm đường (nét đứt); bề ngang theo box, chiều dài theo cỡ trung bình của lớp. Bấm box trên BEV để chọn, sửa
box trên ảnh thì BEV cập nhật ngay; kéo để di chuyển, lăn chuột để zoom.

Phím tắt: `↑/↓` chọn object · `K` keep · `D` delete · `C` đổi lớp · `E` sửa box · `B` vẽ box mới ·
`A` approve low-risk · `Enter` approve frame · `N/P` frame kế/trước · `L` LiDAR · `G` GT; chế độ Video thêm
`T` lan truyền · `Space` phát/dừng.

## QC (kiểm soát chất lượng)

QA Agent chỉ chấm box **detector** lúc pipeline chạy. QC lo phần còn lại: nhãn người sửa, phần duyệt theo lô, và
file nhãn từ ngoài. Ba luồng:

| Luồng | Khi nào | Làm gì |
|---|---|---|
| **QC nhãn cuối** | Sau mỗi thao tác của người, khi approve frame, khi xuất | Box người vẽ / đã sửa / đổi lớp được kiểm lại hình học + LiDAR (`NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH`, `ASPECT_RATIO_ABNORMAL`, `BOX_TOO_LARGE`); cả frame được kiểm box trùng (`DUPLICATE_BOX`), box khác lớp chồng khít (`OVERLAP_CROSS_CLASS`), box hỏng / ra ngoài ảnh / lớp lạ (error). Còn lỗi thì **không approve được**: sửa (có nút áp đề xuất), hoặc xác nhận cảnh báo "đã kiểm, giữ nguyên" kèm lý do. Sửa box sau khi xác nhận thì cảnh báo hiện lại |
| **Audit ngẫu nhiên** | Tab QC | Object duyệt theo lô không ai xem riêng, nên tỉ lệ sửa nhóm low trong Metrics luôn ≈ 0. Audit lấy mẫu ngẫu nhiên (seed lưu lại) cho người xem từng cái, ước lượng tỉ lệ lỗi còn lọt bằng khoảng Wilson 95%; mẫu frame để ước lượng vật bị sót. Mẫu sai → frame mở lại, object quay về chờ duyệt (`AUDIT_FAILED`) |
| **Quick Check** | Tab QC, `POST /qc/quick-check`, `python -m src.cli quick-check` | Kiểm nhanh file nhãn (COCO `.json`, `labels.jsonl`, `{"frames": [...]}`) của người gán thuê ngoài / tool khác / bản export cũ, **không ghi gì**. Frame có trong workspace được đối chiếu thêm LiDAR và detection đã lưu: `POSSIBLY_MISSING` (detector thấy ổn định mà file không có, kèm box đề xuất), `MODEL_DISAGREES` |

**Checklist READY** (tab QC, `GET /qc/report`, `python -m src.cli qc-report`): có frame đã approve · nhãn cuối
không còn lỗi QC mở · mỗi track giữ một lớp · correction log khớp nhãn đang lưu · audit object và audit frame đạt.
Frame còn lỗi QC bị bỏ khi xuất; tick *Chỉ xuất khi READY* để chặn xuất khi checklist chưa đạt. Bản export có thêm
`qa_report.md`, `qc_log.jsonl` (mọi xác nhận / kết quả audit), và `manifest.json` ghi trạng thái READY, nguồn gốc nhãn
(model / track / human / propagated, số object duyệt theo lô), SHA256 từng file, băm config và commit.

Đo trên 79 keyframe nuScenes thật trong workspace: QC trên output detector chưa sửa báo **0** lỗi (không làm ồn luồng
review); Quick Check GT nuScenes (nhãn đúng, 1140 box) báo nhầm ~2% nhãn (`NO_LIDAR_SUPPORT` 1.0%,
`SIZE_DEPTH_MISMATCH` 0.5%), chạy ~4 ms/frame; Quick Check bản export của chính hệ thống: 0 lỗi mở.
Ngưỡng nằm ở mục `qc:` trong [configs/autolabel.yaml](configs/autolabel.yaml).
## Phần 3D

Mô hình 3D đã huấn luyện sẵn trên nuScenes (MMDetection3D: PointPillars, SSN, CenterPoint, FCOS3D, PGD, BEVFusion)
sinh pre-label box 3D; QA Agent kiểm chứng từng box bằng 6 camera + LiDAR, không dùng nhãn gốc
(`src/services/verify3d.py`, port từ `scripts/verify_objects.py` của nhóm 3D):

| Kết luận | Mức | Ý nghĩa |
|---|---|---|
| `DUNG` | low | detector 2D thấy đúng lớp ở đúng chỗ box chiếu xuống |
| `DUNG VAT, BOX LECH` | medium | cùng lớp nhưng box chiếu lệch (IoU thấp) |
| `CAMERA KHONG XAC NHAN` | medium | ảnh rõ, detector không thấy, nhưng có điểm LiDAR trong box |
| `CHUA DU THONG TIN` | medium | bị che / tối / quá xa, không kết luận được |
| `SAI LOP` | high | detector thấy vật khác lớp ở đúng chỗ đó |
| `NGHI BAO NHAM` | high | ảnh rõ, không có vật, không có điểm LiDAR |

```bash
# 1. Suy luận + chấm mAP/NDS các mô hình trên máy có GPU (môi trường riêng, xem tools3d/README.md)
.venv-mm3d\Scripts\python tools3d\run3d.py all --dataroot ..\v1.0-trainval
#    thêm --tta và "ensemble" (gộp 4 mô hình LiDAR + tinh chỉnh theo track): dev mAP 0.537 -> 0.644, test 0.578 -> 0.668
# 2. Kiểm chứng bằng camera, tạo frame 3D cho UI (môi trường chính)
python -m src.cli label3d --model centerpoint_voxel
python -m src.cli evaluate3d --model centerpoint_voxel   # kết luận kiểm chứng so với nhãn gốc
```

UI 3D: khung 3D (xoay/zoom, điểm LiDAR màu theo độ cao, box màu theo mức rủi ro), BEV nhìn từ trên, ảnh camera có
box chiếu xuống (camera tốt nhất tự chọn, `1`–`6` đổi camera). Hàng đợi frame sắp theo rủi ro; từng box: Keep / Delete /
đổi lớp / sửa box; *Approve low-risk* rồi approve frame; tab Metrics có số liệu 3D và bảng so sánh mô hình; Export ra
định dạng kết quả nuScenes (hệ toàn cục).

- **Ảnh BEV** (`I`): 6 camera ghép thành ảnh nhìn từ trên xuống bằng homography mặt đường
  (`src/services/bev.py`), trải dưới point cloud. Thấy được vạch kẻ đường, lề, vị trí xe trên làn quanh cả xe thay vì
  nhìn từng camera một. Mặt đường ghép thêm từ ±4 keyframe cùng scene (nối bằng ego pose), mỗi ô lấy từ lần camera
  nhìn gần nhất: chỗ bị xe che được lấp, ảnh ở xa hết nhoè. Chỗ mặt đường bị vật cao che ở mọi lần nhìn để trống cho
  point cloud hiện ra. Lần đầu mở một frame mất vài giây (sau đó lấy từ cache).
- **Vẽ thêm box** (`B`, cho vật mô hình bỏ sót): chọn lớp, kéo trên mặt đường từ đuôi tới đầu vật (hoặc bấm một điểm để
  đặt box cỡ trung bình của lớp, cùng hướng với xe gần nhất). Box tự đặt đáy lên mặt đường và lấy chiều cao theo điểm
  LiDAR; `F` co khít đám điểm LiDAR (giữ cạnh gần xe khi vật chỉ lộ một mặt). Box hiện ngay trên ảnh camera để so.
- **Sửa box** (`E`): kéo trong box để di chuyển, chấm góc để đổi cỡ, chấm vàng để xoay; hoặc phím `←↑→↓` (theo màn hình),
  `Q/E` xoay, `[ ]` dài, `; '` rộng, `- =` cao, `Shift` bước lớn; hoặc gõ số trong thẻ sửa. `Enter` lưu, `Esc` huỷ.
  Box gốc của mô hình được giữ trong `original_box` và log để đo mô hình lệch bao nhiêu.

Phím: `↑/↓` chọn box · `K` `D` `C` `E` · `B` vẽ box · `A` · `Enter` · `G` GT · `V` đổi khung 3D/BEV · `I` ảnh BEV ·
`L` ẩn/hiện LiDAR · `N/P` frame.

## Kết quả đánh giá

Bảng so sánh mọi phương pháp đã dùng (2D, 3D, có / không dùng thì tăng / giảm bao nhiêu):
[eval/results/bang-so-sanh.md](eval/results/bang-so-sanh.md). Test case thủ công với output thực tế:
[docs/eval-evidence.md](docs/eval-evidence.md). Detector 2D: YOLOE-26-L fine-tune trên nuImages, test 24 scene mAP50
0.366 (zero-shot 0.312). Xem thêm [eval/results/autolabel2d_eval.md](eval/results/autolabel2d_eval.md). Optical flow cho lan truyền và QA
temporal (trước / sau, dev + held-out): [eval/results/temporal/report.md](eval/results/temporal/report.md).

**3D trên tập test** ([eval/results/det3d_heldout.md](eval/results/det3d_heldout.md)): 24 scene val chưa dùng để chọn
cấu hình (957 keyframe), so với nhãn gốc nuScenes. Mô hình đơn tốt nhất CenterPoint voxel mAP 0.578 / NDS 0.655; gộp 4 mô
hình LiDAR + tinh chỉnh theo track (mặc định) **mAP 0.668 / NDS 0.713**. Thử ý tưởng VESPA (box 3D từ box 2D + LiDAR,
hướng theo chuyển động, cỡ theo lớp): không tăng mAP ([eval/results/vespa.md](eval/results/vespa.md)). Thử OC-SORT cho
tracker lan truyền ([eval/results/ocsort.md](eval/results/ocsort.md)): 2D không lợi; 3D giữ track 4 keyframe thay vì 2
(test 24 scene: nhãn lan truyền đúng +1.6%, 94% box đúng vật). Mọi lần tối ưu: PRD, mục "Lịch sử tối ưu".

3D ([eval/compare_3d.ipynb](eval/compare_3d.ipynb)): trọng số có sẵn, 27 scene val nuScenes (1076 keyframe), RTX 4050.
Hai cột cuối là kết quả QA Agent 3D trên 3 scene demo, dùng ngưỡng điểm riêng của từng mô hình.

| Mô hình | Cảm biến | mAP | NDS | s/keyframe | Tự duyệt (đúng) | Bắt box sai |
|---|---|---|---|---|---|---|
| **CenterPoint voxel** (mặc định) | LiDAR | **0.573** | **0.647** | 0.62 | 51% (93%) | 86% |
| CenterPoint pillar | LiDAR | 0.521 | 0.605 | 0.41 | 48% (92%) | 87% |
| SSN | LiDAR | 0.451 | 0.569 | 0.59 | 43% (92%) | 91% |
| PointPillars | LiDAR | 0.470 | 0.562 | 0.85 | 54% (95%) | 88% |
| PGD | Camera | 0.394 | 0.446 | 2.71 | 59% (87%) | 74% |
| FCOS3D | Camera | 0.329 | 0.411 | 2.72 | 53% (80%) | 72% |

## Biến môi trường

Chép `.env.example` thành `.env` (không commit). Mọi biến đều có mặc định trong `src/config.py`, nên không có `.env`
vẫn chạy được bản demo.

| Biến | Mặc định | Dùng để |
|---|---|---|
| `NUSCENES_DATAROOT` | `./v1.0-mini-001` | Thư mục nuScenes (`v1.0-*/`, `samples/`, `sweeps/`) |
| `NUSCENES_VERSION` | `v1.0-mini` | `v1.0-mini` hoặc `v1.0-trainval` |
| `WORKSPACE_DIR` | `./data/workspace` | Workspace của chế độ Ảnh / Video: frame, nhãn, cache detection, correction log |
| `PROJECTS_DIR` | `./data/projects` | Dự án của end-user, mỗi dự án một thư mục con |
| `AUTOLABEL_CONFIG` | `./configs/autolabel.yaml` | Mô hình, ngưỡng, trọng số QA Agent, lan truyền |
| `REVIEWER_NAME` | `annotator` | Tên ghi vào correction log |
| `MM3D_PYTHON` | (tự tìm `.venv-mm3d`) | Python của môi trường MMDetection3D cho bước 3D |
| `AUTH_REQUIRED` | `false` | `true`: phải đăng nhập mới dùng API / UI (tài khoản ở `USERS_FILE`, mặc định `./data/users.json`) |
| `AUTH_OPEN_SIGNUP` | `false` | Cho tự đăng ký sau người đầu tiên (mặc định chỉ vào qua link mời) |
| `PUBLIC_URL` | (lấy từ request) | Gốc của link mời gửi cho người khác, vd `http://192.168.1.10:8000` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | — | Gửi mail mời; chưa có thì UI hiện link để gửi tay |
| `APP_HOST`, `APP_PORT`, `APP_ENV`, `LOG_LEVEL`, `CORS_ORIGINS` | `0.0.0.0`, `8000`, `development`, `INFO`, `http://localhost:3000` | Web server |
| `AI_LOG_SERVER`, `AI_LOG_API_KEY`, `AI_LOG_DIR` | — | Hook ghi log AI khi commit / push (giảng viên cấp key) |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_TRACING_V2` | — | Tuỳ chọn: trace QA Agent lên LangSmith |

`scripts\tasks.ps1 serve -Workspace <thư mục> -Dataroot <nuScenes>` đặt `WORKSPACE_DIR` / `NUSCENES_DATAROOT` cho một
lần chạy, không cần sửa `.env`.

## API

| Method | Path | Mô tả |
|---|---|---|
| GET | `/api/v1/frames?sort=risk&status=` | Hàng đợi frame + số object theo mức risk |
| GET | `/api/v1/frames/{id}` | Frame + object + kết quả QA |
| GET | `/api/v1/frames/{id}/image?offset=` | Ảnh keyframe (0) hoặc sweep (±1, ±2) |
| GET | `/api/v1/frames/{id}/lidar`, `/gt` | Điểm LiDAR đã chiếu, GT 2D |
| GET | `/api/v1/frames/{id}/bev`, `/bev/meta` | Ảnh camera chiếu xuống mặt đường (PNG) và homography / ngoại tham số để đặt box lên BEV |
| POST | `/api/v1/frames/{id}/actions` | `KEEP` / `DELETE` / `CHANGE_CLASS` / `EDIT_BOX` / `ADD_BOX` |
| POST | `/api/v1/frames/{id}/sweeps/{offset}/actions` | Sửa box ở sweep t±n: `KEEP` / `DELETE` / `CHANGE_CLASS` / `EDIT_BOX` / `ADD_BOX` / `RESTORE` (`box_id`), rồi tính lại QA keyframe |
| POST | `/api/v1/frames/{id}/approve-low-risk` | Duyệt theo lô nhóm low |
| POST | `/api/v1/frames/{id}/approve`, `/reopen` | Approve frame (chặn nếu còn object chờ) / mở lại |
| GET | `/api/v1/corrections?frame_id=` | Correction log |
| GET | `/api/v1/metrics` | M1, M4, flag precision/recall, theo nhóm và theo issue |
| POST | `/api/v1/frames/{id}/propagate` | Lan truyền từ frame đã approve sang các keyframe sau còn "auto" |
| GET | `/api/v1/videos` | Video: scene nuScenes + mp4 đã tải lên (tiến độ, số frame đã duyệt / lan truyền) |
| GET | `/api/v1/videos/{id}` | Timeline của một video: danh sách frame kèm thời điểm và trạng thái |
| POST | `/api/v1/videos/upload` | Tải lên mp4 (multipart `file`); cắt frame ngay, auto-label chạy nền |
| POST | `/api/v1/export?require_ready=` | Xuất dataset các frame đã approve và sạch QC (COCO có `segmentation` từ mask sơ bộ; `require_ready=true`: từ chối khi chưa READY) |
| GET | `/api/v1/frames/{id}/qc` | QC nhãn cuối của frame (finding + số lỗi còn mở) |
| POST | `/api/v1/frames/{id}/qc/ack` | Xác nhận "đã kiểm, giữ nguyên" một cảnh báo QC (`key`, `fingerprint`, `note`) |
| GET | `/api/v1/qc/report` | Checklist READY + lỗi còn mở, track đổi lớp, lệch log, audit |
| POST | `/api/v1/qc/quick-check` | Kiểm nhanh file nhãn (multipart `file`), không ghi gì |
| GET / POST | `/api/v1/qc/audit`, `/qc/audit/sample`, `/qc/audit/{id}` | Mẫu audit, lấy mẫu mới (`kind`, `size`, `seed`), ghi kết quả (`ok` / `error`) |
| POST | `/api/v1/frames/{id}/reject` | Reviewer trả lại frame, bắt buộc có lý do (`reason`) |
| POST | `/api/v1/frames/{id}/undo`, `/redo`; GET `/history` | Hoàn tác / làm lại thao tác trên frame (Ctrl+Z / Ctrl+Y) |
| GET/PUT/DELETE | `/api/v1/settings` | Cài đặt chỉnh trên UI của workspace / dự án (`{"values": {"qa.temporal.rescore": "mean", ...}}`) |
| POST/GET | `/api/v1/relabel`, `/api/v1/eval-temporal` | Chạy nền: áp dụng lại cài đặt cho frame chưa mở; so sánh trước / sau optical flow (GET trả tiến độ và kết quả gần nhất) |
| GET | `/api/v1/report.csv?kind=frames\|summary` | Báo cáo CSV: từng frame / số liệu tổng hợp (gồm năng suất) |
| GET | `/api/v1/3d/models`, `/3d/frames?model=` | Mô hình 3D có frame; hàng đợi frame 3D |
| GET | `/api/v1/3d/frames/{m}/{f}` (+ `/points`, `/gt`, `/image/{camera}`) | Frame 3D, point cloud float32 xyzi, GT, ảnh |
| POST | `/api/v1/3d/frames/{m}/{f}/actions`, `/approve-low-risk`, `/approve`, `/reopen` | Duyệt box 3D: `KEEP` / `DELETE` / `CHANGE_CLASS` / `EDIT_BOX` / `ADD_BOX` |
| GET | `/api/v1/3d/frames/{m}/{f}/bev?range=40&res=0.1` | Ảnh BEV ghép 6 camera (PNG RGBA), header `X-Ground-Z` |
| GET | `/api/v1/3d/metrics?model=`, `/3d/corrections`, `/3d/compare` | Số liệu duyệt 3D, log, bảng so sánh mô hình |
| POST | `/api/v1/3d/export?model=` | Xuất box 3D đã duyệt (định dạng nuScenes detection) |
| POST | `/api/v1/3d/export-kitti?model=` | Xuất box 3D đã duyệt dạng KITTI object (ảnh, velodyne, calib, label_2), tải ở `/3d/exports/{file}` |
| POST | `/api/v1/3d/frames/{m}/{f}/reject`, `/undo`, `/redo` | Trả lại / hoàn tác như 2D; `GET /3d/report.csv?model=` |

### Ví dụ gọi API

Server chạy ở `http://localhost:8000` (workspace `data/eval_temporal/ws_dev`). Output dưới đây chép từ lần chạy thật,
rút gọn; xem thêm [docs/eval-evidence.md](docs/eval-evidence.md).

```bash
# 1. Hàng đợi frame, khó nhất trước
curl "http://localhost:8000/api/v1/frames?sort=risk"
# [{"frame_id": "scene-0101_023", "frame_risk": 0.834, "counts": {"low": 18, "medium": 4, "high": 1}, "pending": 23, ...}, ...]

# 2. Chi tiết một frame: box, mức rủi ro, lý do
curl "http://localhost:8000/api/v1/frames/scene-0101_023"
# {"objects": [{"object_id": "23", "label": "traffic_cone", "score": 0.3469, "qa": {"risk": 0.834, "level": "high",
#   "issues": [{"code": "NO_LIDAR_SUPPORT", "message": "Chỉ 1 điểm LiDAR trong box cao 85px (cần ≥ 3)"}, ...]}}, ...]}

# 3. Sửa nhãn: xoá / đổi lớp / thêm box
curl -X POST "http://localhost:8000/api/v1/frames/scene-0101_023/actions" -H "Content-Type: application/json" \
     -d '{"action": "CHANGE_CLASS", "object_id": "18", "label": "barrier"}'
curl -X POST "http://localhost:8000/api/v1/frames/scene-0101_023/actions" -H "Content-Type: application/json" \
     -d '{"action": "ADD_BOX", "label": "traffic_cone", "bbox": [1500, 640, 1560, 760]}'

# 4. Duyệt theo lô nhóm low, rồi approve (409 PENDING_OBJECTS nếu còn box chưa xem)
curl -X POST "http://localhost:8000/api/v1/frames/scene-0101_023/approve-low-risk" -H "Content-Type: application/json" -d "{}"
curl -X POST "http://localhost:8000/api/v1/frames/scene-0101_023/approve" -H "Content-Type: application/json" -d "{}"

# 5. Số liệu duyệt và xuất dataset
curl "http://localhost:8000/api/v1/metrics"
# {"frames": {"total": 119, "approved": 1}, "m4_correction_rate": 0.087, "flag_recall": 1.0, ...}
curl -X POST "http://localhost:8000/api/v1/export" -H "Content-Type: application/json" -d "{}"
# {"export_id": "20261001-052938", "n_frames": 1, "n_objects": 23, "files": ["coco.json", "labels.jsonl", "corrections.jsonl", "manifest.json"]}
```

Trên PowerShell, dùng `curl.exe` thay cho `curl` (hoặc `Invoke-RestMethod`).

## Duyệt, báo cáo, quyền riêng tư

| Yêu cầu (PRD) | Làm thế nào |
|---|---|
| FR-03 làm mờ mặt / biển số | Mọi ảnh gửi tới trình duyệt và ảnh trong file KITTI đều qua `src/services/privacy.py`: biển số bằng detector chuyên dụng (open-image-models, chạy cả ảnh + 4 ô), mặt bằng YOLOE "human face" + vùng đầu của người đủ lớn. Cache ở `workspace/anon/`, ảnh gốc giữ nguyên cho model. Tắt: `privacy.enabled: false` |
| FR-04 mask sơ bộ | Mask của YOLOE-seg lưu thành đa giác trong mỗi object, tô mờ trên ảnh (ô **Mask**), xuất vào `segmentation` của COCO (bỏ khi người sửa box). Ảnh detect trước bản này chưa có mask: xoá `cache/detections/yoloe-*` rồi chạy lại nếu cần |
| FR-06 ngưỡng confidence | Thanh trượt **Score ≥** ở UI 2D và 3D, số box hiển thị đổi ngay |
| FR-09 undo / redo | Nút ↶ ↷, Ctrl+Z / Ctrl+Y; mỗi frame giữ 50 bước (`workspace/history/`), log ghi `UNDO` / `REDO` |
| FR-15 reject kèm lý do | Nút **Trả lại** (phím `R`), lý do bắt buộc; frame lên đầu hàng đợi, không được xuất cho tới khi duyệt lại |
| FR-16 lịch sử | `corrections.jsonl` (từng object) + `events.jsonl` (approve / reject / reopen / undo / redo của từng frame) |
| FR-17 xuất KITTI | Nút **Xuất KITTI** (tab Metrics ở chế độ 3D, trang Dự án); đọc lại được bằng `KittiDB` của nuscenes-devkit |
| FR-19 báo cáo CSV | Tab Metrics: **CSV từng frame**, **CSV tổng hợp** (Excel mở đúng tiếng Việt) |
| Lan truyền 3D | Approve một keyframe 3D (ô **↦ Lan truyền** bật) hoặc phím `T`: box đã duyệt sang các keyframe sau còn chưa mở, bù chuyển động xe + dịch theo vận tốc, lớp / kích thước theo người, box người đã xoá tự xoá. Held-out: ~95% box lan truyền đúng vật ([eval/results/propagation3d.md](eval/results/propagation3d.md)). `POST /3d/frames/{m}/{f}/propagate` |
| Thời gian từng bước | Mỗi frame ghi giây của từng bước (detect, chiếu LiDAR, flow, QA; 3D: detect 6 camera, point cloud, kiểm chứng), thêm nạp model và làm mờ ảnh lần đầu: bảng ở tab **Metrics** (`GET /timing`). Đo từ đầu không dùng cache: `python -m src.cli profile --limit 10` hoặc `scripts\tasks.ps1 profile` |
| Cài đặt trên UI | Tab **⚙ Cài đặt** (cũng mở từ thẻ dự án): optical flow cho lan truyền / QA temporal, tính lại score theo sweep, ngưỡng giữ box, số keyframe lan truyền — lưu theo workspace / dự án (`settings.json`), không phải sửa `configs/autolabel.yaml`. Nút **Áp dụng lại** (frame chưa ai mở, dùng cache detection) và **Chạy đánh giá** (so sánh trước / sau trên frame có GT, bảng ngay trên trang) chạy nền |
| FR-27 năng suất | Tab Metrics: frame/giờ theo người và theo phiên (cách nhau > 30 phút là phiên mới); auto-label frame/giờ theo phiên chạy |

Các việc chạy trên terminal gom trong `scripts\tasks.ps1` (PowerShell): `check`, `install`, `serve`, `test`, `demozip`,
`eval3d`, `label3d`, `eval2d`, `push`. Ví dụ: `powershell -ExecutionPolicy Bypass -File scripts\tasks.ps1 check`.

## Cấu trúc

```
configs/autolabel.yaml     taxonomy, prompt, ngưỡng QA, trọng số risk
src/
  agents/                  QA Agent (LangGraph): state, graph, nodes/{confidence,lidar,temporal,issues,risk}
  services/
    nuscenes_data.py       loader nuScenes, chiếu LiDAR -> ảnh, GT 2D
    detectors/             YOLOE-26, YOLO26, YOLO-World, Grounding DINO, Florence-2, fusion, cache; demo.py (cho demo/test)
    pipeline.py            bước 1 -> 3 (label_keyframe dùng chung cho nuScenes và video tải lên)
    video.py               chế độ Video: cắt mp4, auto-label nền, danh sách/timeline video
    propagation.py         lan truyền: tracker 12Hz, c_prop, ghi vào keyframe đích
    sequence.py            lan truyền cả scene + thí nghiệm keyframe hoàn hảo
    review.py              thao tác review, M4, metrics
    qc/                    QC: checks (nhãn cuối), quick_check, audit (Wilson), report (checklist READY, qa_report.md)
    store.py exporter.py evaluation.py
  api/routes.py            REST API
  web/                     UI review (HTML/JS/CSS)
  cli.py                   python -m src.cli run | evaluate | detect-sweeps | propagate | eval-propagation | qc-report | quick-check
    verify3d.py label3d.py review3d.py eval3d.py   phần 3D: kiểm chứng bằng camera, tạo frame, duyệt, đánh giá
    bev.py                 ảnh BEV bằng homography mặt đường: 1 camera (chế độ Ảnh/Video), ghép 6 camera (3D)
  api/routes3d.py          REST API phần 3D
    projects.py            dự án của end-user: hàng đợi nền, các bước ingest -> 2D -> 3D -> kiểm chứng, tiến độ
    ingest/                nhận dữ liệu: giải nén an toàn, nhận dạng loại, KITTI / LiDAR+camera -> nuScenes (nusc_writer.py)
    export_nusc.py         xuất box 3D đã duyệt thành bảng nuScenes (sample_annotation + instance theo track)
  api/projects_routes.py   REST API dự án (tải lên, tiến độ, chạy lại, xuất, tải về)
  web/projects.html        trang Dự án (mặc định khi mở /)
  cli.py                   python -m src.cli run | evaluate | ... | label3d | evaluate3d
tools3d/                   run3d.py + setup.ps1: suy luận và so sánh mô hình 3D trên GPU (môi trường MMDetection3D riêng)
  refine3d.py              gộp nhiều mô hình 3D + tinh chỉnh theo track (kích thước, vận tốc, hướng, nội suy)
tools2d/                   fine-tune YOLOE-26L trên nuImages + chấm dev/held-out, hướng dẫn chuyển file giữa hai máy
  demo.py                  python -m src.demo: demo không cần GPU/nuScenes
scripts/pack_nuscenes_subset.py   chọn ngẫu nhiên / đóng gói vài scene nuScenes (+ workspace) thành zip nhỏ
scripts/bench_detectors.py        đo tốc độ detector
scripts/simulate_review.py        người duyệt mô phỏng theo GT (đo công duyệt có / không lan truyền)
eval/compare_detectors.ipynb      so sánh detector (matplotlib)
eval/precision_tuning.ipynb       tăng precision không cần train (ngưỡng min_score, ensemble)
eval/review_simulation.ipynb      công duyệt 40 keyframe: có / không lan truyền
eval/compare_3d.ipynb             so sánh mô hình 3D: mAP/NDS, tốc độ, AP từng lớp, P/R theo ngưỡng, kết quả kiểm chứng
tests/                     pytest, dữ liệu tổng hợp (không cần GPU/dataset)
```

## Giới hạn

- nuScenes mini chỉ 10 scene → số liệu mang tính minh hoạ quy trình.
- GT 2D là hộp bao của box 3D chiếu xuống, rộng hơn box sát vật thể → AP@0.7 thấp là bình thường.
- Chưa có: mask SAM2 / sửa mask bằng brush, VLM verifier, đăng nhập/phân vai (web dự án chạy một máy chủ, không đăng nhập).
- Làm mờ dùng detector mở (không phải EgoBlur): biển số xa / nghiêng và mặt nghiêng có thể sót; ảnh BEV ghép từ ảnh gốc
  chưa làm mờ. Lần đầu mở một ảnh chậm thêm vì phải detect (sau đó lấy từ cache).
- Florence-2 không trả confidence nên box của nó nhận score cố định trong config.
- 3D chỉ dùng trọng số có sẵn (chưa fine-tune); FCOS3D/PGD/BEVFusion cần GPU, BEVFusion còn phải biên dịch op CUDA.
  Mô hình 3D học trên LiDAR 32 tia của nuScenes: với LiDAR khác (KITTI 64 tia, cường độ khác thang) độ chính xác giảm.
- Lan truyền chỉ 2D trên một camera (CAM_FRONT hoặc video tải lên); `track_id` đã có sẵn để gắn box 3D vào sau.
- Vật bị che hoàn toàn quá `max_coast_images` ảnh thì track dừng; khi hiện lại nó là object mới cần duyệt. Chạy lại `run --overwrite`
  trên frame "auto" sẽ xoá nhãn lan truyền của frame đó (dựng lại từ cache detection).

Template gốc AI20K (hướng dẫn hook ghi log AI, Technical Guidebook) xem `README_boilerplate.md` và `docs/guide/`.

## License

MIT
