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

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Loader dữ liệu | Tự viết, chỉ numpy | nuscenes-devkit kéo nhiều dependency, pipeline chỉ cần vài bảng |
| Detector mặc định | YOLO-World | Nhanh đủ để chạy cả sweep (5 ảnh/frame) trên GPU 6GB |
| Grounding DINO | bản open-weight tiny | Bản 1.5 Edge chỉ có qua API |
| QA Agent | LangGraph, deterministic | Cần chạy trong batch và test; không có câu hỏi mở nào cần LLM |
| Lưu trữ | File JSON + JSONL | 404 frame, 1 người duyệt; không cần DB cho demo, dễ diff/export |
| UI | HTML/JS thuần do FastAPI phục vụ | Không cần build step, 1 lệnh là chạy |
| GT 2D | Hộp bao 8 đỉnh box 3D chiếu xuống | nuScenes không có box 2D gốc; báo cáo AP@0.5 là chính |

## Chưa làm

- Ẩn danh mặt/biển số (FR-03): EgoBlur cần tải weights có license, chưa tích hợp.
- Mask SAM2, VLM verifier, isotonic calibration (tuần 4 trong PLAN).
- Đăng nhập/phân vai (FR-21): hiện chỉ ghi tên người duyệt vào log.
