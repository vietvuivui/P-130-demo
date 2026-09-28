# Worklog — Team P-130 (AutoLabel 3D)

> Ghi lại tất cả công việc đã làm theo ngày. Ai làm gì, kết quả gì.
> Chi tiết từng lần push (file nào, ảnh hưởng gì, cách kiểm tra) ở [CHANGELOG.md](CHANGELOG.md).

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
