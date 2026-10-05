# Worklog — Team P-130 (AutoLabel 3D)

> Ghi lại tất cả công việc đã làm theo ngày. Ai làm gì, kết quả gì.
> Chi tiết từng lần push (file nào, ảnh hưởng gì, cách kiểm tra) ở [CHANGELOG.md](CHANGELOG.md).

---

## 2026-10-04

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Fine-tune toàn mạng YOLOE-26L trên nuImages (30 epoch, 1280 px, RTX 3090) thành detector 2D mặc định; ô "Mô hình phát hiện 2D" ở tab Cài đặt (Fine-tune full / Fine-tune mini / Original) | ✅ Done | Nhánh `kien`; test 957 keyframe mAP50 0.589 (gốc 0.312, linear probe 0.366) | — |
| Kiên | Chấm lại lan truyền 2D + QA với detector mới (`evaltemporal`), cập nhật report, dọn tài liệu cũ, mở PR `kien` → `main` | ✅ Done | `eval/report/REPORT.md`, `eval/results/bang-so-sanh.md`, `eval/results/temporal/` | — |
| Kiên | Dò lại QA Agent 2D cho detector mới (trọng số risk, ngưỡng, điểm LiDAR); bỏ điểm temporal và đề xuất RECOVERED_BY_TRACK từ detector | ✅ Done | `eval/results/qa_tuning.md`; box sai lọt duyệt theo lô 3621 → 280 trên held-out, 237 test pass | — |
| Kiên | Đo gộp box 3D vào nhãn 2D trên cả 6 camera với detector mới; hạ điểm mạnh hơn cho box chỉ camera không có điểm LiDAR | ✅ Done | `eval/results/lidar2d.md`; test 6 camera mAP50 0.560 → 0.672, box phải sửa 23432 → 15184 | — |

**Tổng kết ngày:** Detector 2D mặc định là bản fine-tune toàn mạng; report và bảng so sánh đã dùng số của bản này. QA Agent 2D đã dò lại theo detector mới.

---

## 2026-10-02

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Theo góp ý mentor: làm việc nhiều người (tài khoản, mời bằng link, vai trò, chia frame, khoá frame), DAM4SAM + BoT-SORT cho lan truyền, TrackEval | ✅ Done | Nhánh `kien-mentor`, 211 test pass | — |
| Kiên | Bấm để chọn vật bằng SAM 2.1 (ONNX), tăng tốc DAM4SAM, một lệnh đo mọi cấu hình tracking (`trackall`), khử box 3D trùng | ✅ Done | Các nhánh `kien-*` đã gộp vào `kien`; `eval/results/tracking.md`, `improve.md` | — |

**Tổng kết ngày:** Sản phẩm dùng được nhiều người trên một dự án; có số HOTA / MOTA / IDF1 cho các cấu hình lan truyền.

---

## 2026-10-01

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Detector 2D mặc định đổi sang YOLOE-26L fine-tune nuImages (linear probe); bảng so sánh mọi phương pháp 2D / 3D | ✅ Done | Nhánh `kien`; test mAP50 0.312 → 0.366; `eval/results/bang-so-sanh.md` | — |
| Danh | Xử lý xung đột sau khi merge `main`, đồng bộ giao diện trang Dự án và Workspace, hoàn thiện Review Panel và thanh điều khiển video | ✅ Done | Nhánh `danh`, PR #16 vào `main` | — |

**Tổng kết ngày:** Có detector fine-tune đầu tiên và bảng so sánh phương pháp; giao diện của Danh đã vào `main`.

---

## 2026-09-30

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Optical flow cho lan truyền nhãn và QA temporal (ý tưởng Deep Feature Flow, mức box); tab ⚙ Cài đặt trên UI | ✅ Done | Nhánh `kien`, 152 test pass; `eval/results/temporal/report.md` | — |

**Tổng kết ngày:** Lan truyền 2D dùng optical flow làm mặc định; người dùng chỉnh được cấu hình trên UI.

---

## 2026-09-29

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | UI 3D: vẽ thêm / sửa box, BEV ghép 6 camera; web cho end-user (trang Dự án, tải lên video / ảnh / nuScenes / KITTI); các FR còn thiếu của PRD (reject kèm lý do, undo / redo, báo cáo CSV, năng suất) | ✅ Done | Nhánh `feat/yoloe-review-sim` và `kien`, 121 test pass | — |
| Huy | QC nhãn cuối (3 luồng); đổi detector mặc định sang YOLOE-26-L, thêm YOLO26-L | ✅ Done | Nhánh `Huy`, 114 test pass | — |
| Danh | Bộ giao diện UI Flow đa trang (Đăng nhập, Dự án, Chọn frame, Xuất dữ liệu) nối vào FastAPI; tái cấu trúc `src/web/` | ✅ Done | Nhánh `danh` | — |

**Tổng kết ngày:** Sản phẩm có luồng end-user đầy đủ từ tải dữ liệu lên tới xuất nhãn, cho cả 2D và 3D.

---

## 2026-09-28

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Thêm detector YOLOE-26 / YOLO26 và notebook so sánh; ngưỡng giữ box + precision / recall / F1; mô phỏng công duyệt; chạy trên dữ liệu chưa gán nhãn; phần 3D (UI three.js, QA Agent 3D, export nuScenes) | ✅ Done | Nhánh `feat/label-propagation`, `feat/yoloe-review-sim`, 91 test pass | — |
| Việt | Notebook Colab: PointPillars (MMDetection3D) tạo box 3D và chấm mAP / NDS, YOLOE-26 kiểm chứng từng box trên ảnh | ✅ Done | Nhánh `viet` | — |

**Tổng kết ngày:** Có nhánh 3D đầu tiên trong sản phẩm và số liệu chọn detector 2D.

---

## 2026-09-27

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Kiên | Lan truyền nhãn 2D trên video (FR-11 → FR-13) + UI + thí nghiệm keyframe hoàn hảo | 🔄 WIP | Nhánh `feat/label-propagation`, 65 test pass; chờ chạy trên nuScenes thật | — |
| Kiên | Chế độ Ảnh / Video trên UI, timeline + kéo-thả frame, tải lên mp4, demo không GPU (`python -m src.demo`), notebook Kaggle + script đóng gói nuScenes | 🔄 WIP | Cùng nhánh, 80 test pass; chờ chạy notebook trên Kaggle | — |

**Tổng kết ngày:** Có lan truyền nhãn từ keyframe đã duyệt sang các keyframe sau, tách thành chế độ Video riêng với timeline; demo chạy được trên máy không GPU. Cần số liệu nuScenes thật để chỉnh ngưỡng.

---

## 2026-09-26

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Huy | Pipeline auto-label 2D + QA Agent + UI review by exception | ✅ Done | Nhánh `Huy`, commit `454bb89`, 48 test pass | — |

**Tổng kết ngày:** MVP 2D chạy đầu cuối trên nuScenes mini (ghi bù từ commit).

---

<!-- Format: copy block trên cho mỗi ngày làm việc -->
