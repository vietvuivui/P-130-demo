# Lan truyền box 3D đã duyệt — trước / sau

Người approve một keyframe 3D → box đã duyệt (lớp, kích thước người chốt) được mang sang các keyframe sau còn "auto"
(`src/services/propagation3d.py`). Box được đưa về hệ toàn cục (bù chuyển động xe bằng ego pose), dịch theo vận tốc
mô hình dự đoán, rồi khớp dự đoán của mô hình ở keyframe sau theo khoảng cách tâm (ngưỡng theo lớp của tracker
CenterPoint). Trước đây phần 3D không có lan truyền: mọi keyframe duyệt lại từ đầu.

## Cách đo (thí nghiệm keyframe hoàn hảo, `src/services/propagation3d_eval.py`)

- Nhãn gốc ở keyframe 0, 5, 10… của mỗi scene đóng vai box người đã duyệt; lan truyền tối đa 10 keyframe (5 s).
- Box ra **đúng** nếu tâm cách nhãn gốc của chính vật đó ≤ 2 m (ngưỡng TP rộng nhất của nuScenes); **đổi ID** nếu trùng
  vật khác; **mất** = vật vẫn có điểm LiDAR mà không có box lan truyền. **Phủ** = đúng / số lần vật xuất hiện.
- Dự đoán: 4 mô hình LiDAR của `run3d.py run` (không TTA), `ensemble` = gộp + tinh chỉnh theo track như sản phẩm,
  ngưỡng điểm 0.3.
- Dev = scene-0035/0097/0101 (chọn cấu hình); held-out = scene-0003/0016/0039/0095 (chỉ báo). Số liệu:
  `propagation3d_7scenes.json`. Chạy trên đủ 27 scene val của máy có dự đoán: `scripts\tasks.ps1 evalprop3d`.

## Kết quả (ensemble)

| Cách | Dev: đúng / box ra | Đổi ID | Sai | Tỉ lệ đúng | Phủ | Held-out: đúng / box ra | Đổi ID | Sai | Tỉ lệ đúng | Phủ |
|---|---|---|---|---|---|---|---|---|---|---|
| chỉ bù chuyển động xe | 3650 / 3980 | 182 | 148 | 0.917 | 0.588 | 3031 / 3283 | 98 | 154 | 0.923 | 0.562 |
| + vận tốc | 4049 / 4343 | 143 | 151 | 0.932 | 0.653 | **3475** / 3706 | 67 | 164 | 0.938 | **0.645** |
| **+ vận tốc, ngưỡng ×0.5, chịu mất 2 keyframe** (mặc định) | **4074** / 4311 | **106** | **131** | **0.945** | **0.656** | 3339 / 3514 | **36** | **139** | **0.950** | 0.619 |

CenterPoint voxel một mình cho kết quả cùng chiều (held-out: 3064 → 3529 → 3234 nhãn đúng; đổi ID 115 → 103 → 48).

- Dịch theo vận tốc là phần quan trọng: +11% (dev) / +15% (held-out) nhãn đúng so với chỉ bù chuyển động xe.
- Cấu hình mặc định (chọn trên dev, nơi nó hơn ở mọi cột) trên held-out đổi 4% nhãn đúng lấy −46% đổi ID và −24% lỗi
  (đổi ID + sai): ~95% box lan truyền đúng vật.
- Với mỗi vật đã duyệt ở keyframe đầu, khoảng 62–65% lần nó xuất hiện ở 10 keyframe sau đã có sẵn box đúng với lớp và
  kích thước người chốt; mô hình tự thấy đúng lớp ở ~71% — phần còn lại vẫn phải duyệt như trước.
