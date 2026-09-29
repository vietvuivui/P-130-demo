# AutoLabel 2D — Auto-label nuScenes + QA Agent đa tín hiệu + Review by exception

> Gán nhãn 2D cho xe tự lái tốn thời gian vì người phải soát từng box → model open-vocab tự sinh box,
> QA Agent kiểm chứng chéo bằng LiDAR và các frame camera lân cận để chấm rủi ro, người chỉ tập trung
> vào số ít box rủi ro cao và duyệt theo lô phần còn lại.

![Workflow](docs/workflow.png)

## Workflow

| Bước | Làm gì | Code |
|---|---|---|
| 1. Input | CAM_FRONT keyframe, sweep t-2..t+2 (12Hz), LIDAR_TOP, calibration + ego pose từ nuScenes v1.0-mini | `src/services/nuscenes_data.py` |
| 2. 2D Detection | Mặc định **YOLOE-26-L** (open-vocab, đủ 10 lớp bằng prompt, không train); chọn được thêm YOLO26-L (COCO), YOLO-World, Grounding DINO, Florence-2 — nhiều model thì fuse theo lớp mỗi model phủ | `src/services/detectors/` |
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

## Quick Start (nuScenes thật)

Yêu cầu: Python 3.11, GPU NVIDIA (chạy được trên RTX 3060 6GB; CPU cũng chạy nhưng chậm),
nuScenes v1.0-mini giải nén vào `./v1.0-mini-001` (hoặc đặt `NUSCENES_DATAROOT` trong `.env`).

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-ml.txt
python scripts/download_weights.py   # YOLOE-26-L + text encoder MobileCLIP2 -> weights/ (kiểm SHA256)

# 1-3. Auto-label + QA Agent (detection được cache theo model, chạy lại rất nhanh)
python -m src.cli run --limit 40                 # 40 keyframe đầu
python -m src.cli run                            # cả 404 keyframe
python -m src.cli run --detectors yolo_world --overwrite   # đổi model (detection cũ vẫn trong cache)

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

Máy chưa tải được nuScenes: chạy notebook [demo/kaggle_nuscenes_run.ipynb](demo/kaggle_nuscenes_run.ipynb) trên
Kaggle (dataset có sẵn ở đó, GPU miễn phí). Notebook auto-label 2 scene rồi đóng gói ảnh CAM_FRONT + bảng nuScenes
đã lọc + workspace thành một file zip ~100 MB; tải về, giải nén vào thư mục repo, `uvicorn src.main:app`.

## UI

Thanh trên cùng có hai chế độ:

- **🖼 Ảnh** — duyệt từng frame như MVP: hàng đợi sắp theo risk, review by exception. Không lan truyền.
- **🎞 Video** — danh sách video (mỗi scene nuScenes là một video; mp4 tải lên bằng nút *⬆ Tải lên mp4*),
  timeline riêng cho video đang mở, lan truyền nhãn giữa các frame.

Chế độ Ảnh:

- **Hàng đợi frame** sắp theo frame risk (khó nhất lên đầu), lọc theo trạng thái.
- **Canvas**: box tô màu theo risk (xanh/vàng/đỏ, luôn có nhãn chữ), box `RECOVERED_BY_TRACK` nét đứt,
  bật overlay điểm LiDAR (màu theo độ sâu) và GT để đối chiếu.
  Box GT (nét đứt trắng, chữ `GT <lớp>`) = box 3D của nuScenes chiếu xuống ảnh (hình chữ nhật bao 8 đỉnh) nên thường
  rộng hơn box detector; box GT mờ không chữ = vật hiển thị 0–40% hoặc không có điểm LiDAR/radar, bỏ qua khi đánh giá.
- **Zoom**: lăn chuột trên ảnh (phóng quanh con trỏ), `+`/`−`, `0` về vừa khung, hoặc nút `− 100% +`; khi phóng to
  kéo vùng trống để di chuyển ảnh. Vẽ/sửa box vẫn đúng toạ độ ở mọi mức zoom.
- **Temporal strip** t-2 … t+2: xem object đang chọn có/không ở từng sweep; click để xem sweep đó.
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

## Kết quả đánh giá

Xem [eval/results/autolabel2d_eval.md](eval/results/autolabel2d_eval.md).

## API

| Method | Path | Mô tả |
|---|---|---|
| GET | `/api/v1/frames?sort=risk&status=` | Hàng đợi frame + số object theo mức risk |
| GET | `/api/v1/frames/{id}` | Frame + object + kết quả QA |
| GET | `/api/v1/frames/{id}/image?offset=` | Ảnh keyframe (0) hoặc sweep (±1, ±2) |
| GET | `/api/v1/frames/{id}/lidar`, `/gt` | Điểm LiDAR đã chiếu, GT 2D |
| POST | `/api/v1/frames/{id}/actions` | `KEEP` / `DELETE` / `CHANGE_CLASS` / `EDIT_BOX` / `ADD_BOX` |
| POST | `/api/v1/frames/{id}/approve-low-risk` | Duyệt theo lô nhóm low |
| POST | `/api/v1/frames/{id}/approve`, `/reopen` | Approve frame (chặn nếu còn object chờ) / mở lại |
| GET | `/api/v1/corrections?frame_id=` | Correction log |
| GET | `/api/v1/metrics` | M1, M4, flag precision/recall, theo nhóm và theo issue |
| POST | `/api/v1/frames/{id}/propagate` | Lan truyền từ frame đã approve sang các keyframe sau còn "auto" |
| GET | `/api/v1/videos` | Video: scene nuScenes + mp4 đã tải lên (tiến độ, số frame đã duyệt / lan truyền) |
| GET | `/api/v1/videos/{id}` | Timeline của một video: danh sách frame kèm thời điểm và trạng thái |
| POST | `/api/v1/videos/upload` | Tải lên mp4 (multipart `file`); cắt frame ngay, auto-label chạy nền |
| POST | `/api/v1/export?require_ready=` | Xuất dataset các frame đã approve và sạch QC (`require_ready=true`: từ chối khi chưa READY) |
| GET | `/api/v1/frames/{id}/qc` | QC nhãn cuối của frame (finding + số lỗi còn mở) |
| POST | `/api/v1/frames/{id}/qc/ack` | Xác nhận "đã kiểm, giữ nguyên" một cảnh báo QC (`key`, `fingerprint`, `note`) |
| GET | `/api/v1/qc/report` | Checklist READY + lỗi còn mở, track đổi lớp, lệch log, audit |
| POST | `/api/v1/qc/quick-check` | Kiểm nhanh file nhãn (multipart `file`), không ghi gì |
| GET / POST | `/api/v1/qc/audit`, `/qc/audit/sample`, `/qc/audit/{id}` | Mẫu audit, lấy mẫu mới (`kind`, `size`, `seed`), ghi kết quả (`ok` / `error`) |

## Cấu trúc

```
configs/autolabel.yaml     taxonomy, prompt, ngưỡng QA, trọng số risk
src/
  agents/                  QA Agent (LangGraph): state, graph, nodes/{confidence,lidar,temporal,issues,risk}
  services/
    nuscenes_data.py       loader nuScenes, chiếu LiDAR -> ảnh, GT 2D
    detectors/             YOLOE-26 (mặc định), YOLO26, YOLO-World, Grounding DINO, Florence-2, fusion, cache;
                           demo.py (theo màu, cho demo/test)
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
  demo.py                  python -m src.demo: demo không cần GPU/nuScenes
scripts/pack_nuscenes_subset.py   đóng gói vài scene nuScenes + workspace thành zip (dùng trên Kaggle)
demo/kaggle_nuscenes_run.ipynb    chạy auto-label trên Kaggle rồi tải kết quả về máy
tests/                     pytest, dữ liệu tổng hợp (không cần GPU/dataset)
```

## Giới hạn

- nuScenes mini chỉ 10 scene → số liệu mang tính minh hoạ quy trình.
- GT 2D là hộp bao của box 3D chiếu xuống, rộng hơn box sát vật thể → AP@0.7 thấp là bình thường.
- Chưa có: ẩn danh mặt/biển số (EgoBlur), mask SAM2, VLM verifier, đăng nhập/phân vai.
- Florence-2 không trả confidence nên box của nó nhận score cố định trong config.
- Lan truyền chỉ 2D trên một camera (CAM_FRONT hoặc video tải lên); `track_id` đã có sẵn để gắn box 3D vào sau.
- Vật bị che hoàn toàn quá `max_coast_images` ảnh thì track dừng; khi hiện lại nó là object mới cần duyệt. Chạy lại `run --overwrite`
  trên frame "auto" sẽ xoá nhãn lan truyền của frame đó (dựng lại từ cache detection).

Template gốc AI20K (hướng dẫn hook ghi log AI, Technical Guidebook) xem `README_boilerplate.md` và `docs/guide/`.

## License

MIT
