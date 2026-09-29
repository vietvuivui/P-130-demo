# tools3d — chạy và so sánh các mô hình 3D trên máy có GPU

Phần 3D của sản phẩm dùng mô hình đã huấn luyện sẵn trên nuScenes (MMDetection3D) để sinh pre-label box 3D, rồi
QA Agent kiểm chứng từng box bằng 6 camera (`src/services/verify3d.py`). Thư mục này chạy bước suy luận, cần GPU
NVIDIA, trong một môi trường Python riêng để không đụng tới môi trường chính của repo.

## Mô hình

| Tên (`-m`) | Mô hình | Cảm biến | mAP / NDS công bố (val đầy đủ) |
|---|---|---|---|
| `pointpillars` | PointPillars FPN (mốc so sánh, nhóm đang dùng) | LiDAR | 39.7 / 53.2 |
| `ssn` | SSN SECFPN | LiDAR | 40.9 / 54.4 |
| `ssn_regnet` | SSN RegNet-400MF (tuỳ chọn) | LiDAR | 46.7 / 58.2 |
| `centerpoint_pillar` | CenterPoint pillar 0.2 | LiDAR | 48.7 / 59.6 |
| `centerpoint_voxel` | CenterPoint voxel 0.075 (backbone SECOND) | LiDAR | 56.5 / 65.2 |
| `fcos3d` | FCOS3D | 6 camera | 32.1 / 39.3 |
| `pgd` | PGD | 6 camera | 35.8 / 42.5 |
| `bevfusion` | BEVFusion (tuỳ chọn, xem dưới) | LiDAR + camera | 68.6 / 71.4 |

SECOND không có trọng số nuScenes trong MMDetection3D (chỉ có KITTI / Waymo); CenterPoint voxel dùng chính backbone
SECOND nên đại diện cho họ SECOND.

## Cài (một lần, ~10 phút)

```powershell
cd <repo P-130>
powershell -ExecutionPolicy Bypass -File tools3d\setup.ps1
```

Tạo `.venv-mm3d` (Python 3.10): PyTorch 2.1.2 + CUDA 11.8, mmcv 2.1.0, mmdet 3.3.0, mmdet3d 1.4.0. Cuối cùng in
tên GPU và phiên bản; báo lỗi thì dán cho nhóm.

## Chạy

```powershell
.venv-mm3d\Scripts\python tools3d\run3d.py all --dataroot ..\v1.0-trainval
```

- Tự tìm các scene thuộc **tập val** có đủ file trên máy (LiDAR keyframe + 10 lần quét trước + 6 camera). Mọi trọng
  số ở đây đều học trên tập train, chấm trên train sẽ bị ảo.
- Tải trọng số (~20–200 MB mỗi mô hình) vào `tools3d/work/ckpts/`, suy luận, chấm mAP / NDS bằng bộ chấm chính thức
  của nuScenes trên đúng các scene đã chạy.
- Một mô hình lỗi (thiếu op, hết VRAM) không chặn các mô hình còn lại; lỗi được ghi vào bảng.
- Thời gian trên RTX 4050 6 GB: vài phút mỗi mô hình LiDAR, 10–20 phút mỗi mô hình camera. Tắt game / ứng dụng
  chiếm VRAM trước khi chạy.
- Chạy nhanh để thử: thêm `--max-scenes 4`, hoặc chọn mô hình `-m pointpillars centerpoint_pillar`.

### Thêm dữ liệu (vd. v1.0-mini)

Bảng của `v1.0-trainval` đã chứa cả 850 scene, chỉ thiếu file. Chép file các scene **val** từ thư mục khác vào
dataroot là lần chạy sau tự dùng thêm các scene đó (mini có 4 scene val: 0103, 0553, 0796, 0916; 6 scene còn lại thuộc
train nên bỏ qua):

```powershell
.venv-mm3d\Scripts\python tools3d\run3d.py add-data --dataroot ..\v1.0-trainval --src ..\v1.0-mini
```

Kết quả cần gửi lại cho nhóm (nhỏ, đưa vào git được): cả thư mục `eval/results/det3d/`

- `summary.json`: bảng so sánh (mAP, NDS, AP từng lớp, sai số, precision/recall theo ngưỡng, thời gian, VRAM)
- `<mô hình>/ui_preds.json`: dự đoán của 3 scene demo (scene-0035, scene-0097, scene-0101) để đưa vào UI 3D

File dự đoán đầy đủ và trọng số nằm trong `tools3d/work/` (đã gitignore).

## Tăng độ chính xác không cần huấn luyện: ensemble + tinh chỉnh theo track + TTA

`eval` tự thêm hai "mô hình" dẫn xuất, tính từ dự đoán đã có (không suy luận lại):

| Tên | Cách làm | File |
|---|---|---|
| `centerpoint_voxel_track` | CenterPoint voxel + tinh chỉnh theo track | `tools3d/refine3d.py` |
| `ensemble` | gộp 4 mô hình LiDAR (PointPillars, SSN, CenterPoint pillar / voxel) rồi tinh chỉnh theo track | `tools3d/refine3d.py` |

**Gộp.** Gom box cùng lớp theo khoảng cách tâm, với bán kính riêng từng lớp. Điểm của box gộp là điểm trung bình nhân
với tỉ lệ mô hình đồng ý: box chỉ một mô hình thấy bị hạ điểm.

**Tinh chỉnh theo track** (kiểu "auto-label offboard": được nhìn cả trước lẫn sau):

- nối box qua các keyframe bằng vận tốc dự đoán;
- mỗi track dùng một kích thước (trung vị);
- tính lại vận tốc bằng sai phân trung tâm, nhưng chỉ khi khớp với vận tốc của mô hình;
- sửa hướng bị lật 180°;
- trộn điểm của track;
- nội suy keyframe bị sót ở giữa track, với điểm thấp để người duyệt xem lại;
- gán `tracking_id`, để xuất nuScenes nối được instance.

**`--tta`** chạy thêm 3 lượt lật trục (x, y, cả hai) cho mỗi mô hình LiDAR. Mỗi lượt được lật ngược kết quả về trước
khi gộp. Chậm hơn khoảng 4 lần.

Kết quả trên 3 scene dev (scene-0035/0097/0101). Tham số lấy theo mặc định của CenterPoint / bài báo, không dò trên dữ
liệu chấm:

| Cấu hình | mAP | NDS |
|---|---|---|
| CenterPoint voxel | 0.537 | 0.560 |
| + tinh chỉnh theo track | 0.573 | 0.583 |
| gộp 4 mô hình LiDAR | 0.617 | 0.621 |
| **gộp 4 mô hình + track** (`ensemble`) | **0.644** | **0.643** |
| thêm 2 mô hình camera vào ensemble | 0.639 | — (không dùng) |
| CenterPoint pillar, scene-0035 (4 sweep), + TTA lật | 0.433 → 0.450 | 0.478 → 0.495 |

Những gì đã thử nhưng không dùng:

- **Chấm lại điểm bằng camera** (hồi quy logistic, kiểm tra chéo bỏ từng scene): mAP giảm từ 0.644 xuống 0.58–0.61.
- **Đổi / xoá box theo detector 2D:** sửa đúng 14 box nhưng làm hỏng 39.

**Con số quyết định phải là held-out**, tức 24 scene val còn lại mà khi làm chưa hề nhìn. Chạy trên máy GPU:

```powershell
.venv-mm3d\Scripts\python tools3d\run3d.py run --dataroot ..\v1.0-trainval -m pointpillars ssn centerpoint_pillar centerpoint_voxel --tta
.venv-mm3d\Scripts\python tools3d\run3d.py eval --dataroot ..\v1.0-trainval
```

`eval/results/det3d/summary.json` có thêm `splits.dev` / `splits.heldout` cho từng mô hình. Nếu `ensemble` hơn
`centerpoint_voxel` trên held-out, dự án của end-user sẽ dùng ensemble.

### Dữ liệu chưa gán nhãn (dự án của end-user)

```powershell
.venv-mm3d\Scripts\python tools3d\run3d.py predict --dataroot <dataset> --version v1.0-custom --out preds.json [--tta]
```

Web gọi lệnh này ở bước "Dự đoán 3D" của dự án: suy luận 4 mô hình LiDAR, gộp, tinh chỉnh theo track, rồi ghi dự
đoán mọi keyframe. Không cần thư mục `maps/`, không cần nhãn.

## Đưa vào sản phẩm

Trong môi trường chính của repo, với `NUSCENES_DATAROOT` trỏ tới dữ liệu có 3 scene demo:

```powershell
python -m src.cli label3d --model centerpoint_voxel     # kiểm chứng bằng camera, tạo frame 3D để duyệt
python -m src.cli evaluate3d                            # kết luận kiểm chứng so với nhãn gốc
uvicorn src.main:app                                    # chọn 🧊 3D trên UI
```

## BEVFusion (tuỳ chọn)

Không có trong gói pip của mmdet3d; phải lấy mã nguồn và biên dịch op CUDA riêng, cần **CUDA Toolkit 11.8** và
**Visual Studio Build Tools 2019/2022 (C++)**. Nếu biên dịch lỗi trên Windows, chạy trên máy Linux / máy 3090.

```powershell
git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git ..\mmdetection3d
cd ..\mmdetection3d
..\P-130\.venv-mm3d\Scripts\python projects\BEVFusion\setup.py develop
cd ..\P-130
.venv-mm3d\Scripts\python tools3d\run3d.py run -m bevfusion --dataroot ..\v1.0-trainval --mmdet3d-repo ..\mmdetection3d
.venv-mm3d\Scripts\python tools3d\run3d.py eval --dataroot ..\v1.0-trainval
```
