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
