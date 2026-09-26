# AutoLabel 2D — Auto-label nuScenes + QA Agent đa tín hiệu + Review by exception

> Gán nhãn 2D cho xe tự lái tốn thời gian vì người phải soát từng box → model open-vocab tự sinh box,
> QA Agent kiểm chứng chéo bằng LiDAR và các frame camera lân cận để chấm rủi ro, người chỉ tập trung
> vào số ít box rủi ro cao và duyệt theo lô phần còn lại.

![Workflow](docs/workflow.png)

## Workflow

| Bước | Làm gì | Code |
|---|---|---|
| 1. Input | CAM_FRONT keyframe, sweep t-2..t+2 (12Hz), LIDAR_TOP, calibration + ego pose từ nuScenes v1.0-mini | `src/services/nuscenes_data.py` |
| 2. 2D Detection | YOLO-World (mặc định), Grounding DINO, Florence-2 — open-vocab, không train; nhiều model thì fuse | `src/services/detectors/` |
| 3. QA Agent | LangGraph: 3.1 confidence · 3.2 LiDAR support · 3.3 temporal (song song) → 3.4 issue → 3.5 risk | `src/agents/` |
| 4. Review by exception | Low risk duyệt theo lô, high risk xem chi tiết: Keep / Delete / Change class / Sửa box / Add box | `src/web/`, `src/services/review.py` |
| 5. Correction log | Mỗi thao tác ghi 1 dòng JSONL: dự đoán, risk, issue, hành động, kết quả cuối | `data/workspace/corrections.jsonl` |
| 6. Dataset | Chỉ frame đã approve: COCO + JSONL + log + manifest | `src/services/exporter.py` |

**Issue code:** `LOW_CONFIDENCE`, `CLASS_CONFLICT`, `NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH`, `FLICKER`,
`RECOVERED_BY_TRACK`, `BOX_TOO_LARGE`, `ASPECT_RATIO_ABNORMAL`.

**Risk:** `risk = w1(1 − score) + w2·lidar + w3·temporal + w4·geometric` → low < 0.30 ≤ medium < 0.60 ≤ high.
Object có issue luôn ≥ 0.30 nên không bao giờ bị duyệt theo lô. Mọi ngưỡng và trọng số nằm trong
[configs/autolabel.yaml](configs/autolabel.yaml). Chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

Yêu cầu: Python 3.11, GPU NVIDIA (chạy được trên RTX 3060 6GB; CPU cũng chạy nhưng chậm),
nuScenes v1.0-mini giải nén vào `./v1.0-mini-001` (hoặc đặt `NUSCENES_DATAROOT` trong `.env`).

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-ml.txt
pip install git+https://github.com/ultralytics/CLIP.git  # text encoder cho YOLO-World

# 1-3. Auto-label + QA Agent (tải weights lần đầu; detection được cache, chạy lại rất nhanh)
python -m src.cli run --limit 40                 # 40 keyframe đầu
python -m src.cli run                            # cả 404 keyframe
python -m src.cli run --detectors yolo_world grounding_dino --overwrite   # ensemble

# Đánh giá so với GT (mAP + flag recall/precision) -> eval/results/autolabel2d_eval.md
python -m src.cli evaluate

# 4-6. UI review
uvicorn src.main:app --port 8000
# mở http://localhost:8000
```

Chỉ chạy UI / test (không cần GPU): `pip install -r requirements.txt`, rồi `pytest` hoặc `uvicorn`.

## UI

- **Hàng đợi frame** sắp theo frame risk (khó nhất lên đầu), lọc theo trạng thái.
- **Canvas**: box tô màu theo risk (xanh/vàng/đỏ, luôn có nhãn chữ), box `RECOVERED_BY_TRACK` nét đứt,
  bật overlay điểm LiDAR (màu theo độ sâu) và GT để đối chiếu.
- **Temporal strip** t-2 … t+2: xem object đang chọn có/không ở từng sweep; click để xem sweep đó.
- **Panel risk**: High (card chi tiết + crop + issue + giải thích), Medium, Low (thu gọn + *Approve all low-risk*).
- **Correction log** và **Metrics & Export**: M4 (tỉ lệ nhãn phải sửa), M1 (thời gian/frame),
  flag precision/recall của agent tính từ thao tác thật của người, precision của từng issue code.

Phím tắt: `↑/↓` chọn object · `K` keep · `D` delete · `C` đổi lớp · `E` sửa box · `B` vẽ box mới ·
`A` approve low-risk · `Enter` approve frame · `N/P` frame kế/trước · `L` LiDAR · `G` GT.

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
| POST | `/api/v1/export` | Xuất dataset các frame đã approve |

## Cấu trúc

```
configs/autolabel.yaml     taxonomy, prompt, ngưỡng QA, trọng số risk
src/
  agents/                  QA Agent (LangGraph): state, graph, nodes/{confidence,lidar,temporal,issues,risk}
  services/
    nuscenes_data.py       loader nuScenes, chiếu LiDAR -> ảnh, GT 2D
    detectors/             YOLO-World, Grounding DINO, Florence-2, fusion, cache
    pipeline.py            bước 1 -> 3
    review.py              thao tác review, M4, metrics
    store.py exporter.py evaluation.py
  api/routes.py            REST API
  web/                     UI review (HTML/JS/CSS)
  cli.py                   python -m src.cli run | evaluate
tests/                     pytest, dữ liệu tổng hợp (không cần GPU/dataset)
```

## Giới hạn

- nuScenes mini chỉ 10 scene → số liệu mang tính minh hoạ quy trình.
- GT 2D là hộp bao của box 3D chiếu xuống, rộng hơn box sát vật thể → AP@0.7 thấp là bình thường.
- Chưa có: ẩn danh mặt/biển số (EgoBlur), mask SAM2, VLM verifier, đăng nhập/phân vai.
- Florence-2 không trả confidence nên box của nó nhận score cố định trong config.

Template gốc AI20K (hướng dẫn hook ghi log AI, Technical Guidebook) xem `README_boilerplate.md` và `docs/guide/`.

## License

MIT
