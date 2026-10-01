# Cài môi trường riêng cho các mô hình 3D (MMDetection3D) trên Windows có GPU NVIDIA.
# Chạy từ thư mục gốc của repo:  powershell -ExecutionPolicy Bypass -File tools3d\setup.ps1
#
# Tạo .venv-mm3d (Python 3.10, không đụng tới môi trường chính của repo) với tổ hợp đã kiểm chứng, có sẵn bản dựng:
#   PyTorch 2.1.2 + CUDA 11.8, mmcv 2.1.0, mmdet 3.3.0, mmdet3d 1.4.0  (hỗ trợ card RTX 20/30/40)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Tìm Python thật (bỏ qua bản giả của Microsoft Store)
$py = $null
foreach ($c in @("py", "python", "python3")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") { $py = $cmd.Source; break }
}
if (-not $py) { throw "Không thấy Python. Cài Python 3.10+ từ python.org rồi chạy lại." }
if (-not (Test-Path ".venv-mm3d\Scripts\python.exe")) {
    # py.exe có thể chọn Python khác nhau giữa các lần gọi -> chốt đường dẫn Python gốc (ngoài mọi venv) một lần
    $base = (& $py -c "import sys; print(getattr(sys, '_base_executable', sys.executable))" | Select-Object -Last 1).Trim()
    Write-Host ">> Dùng $base để cài uv"
    & $base -m pip install --user --quiet uv
    if ($LASTEXITCODE -ne 0) { throw "Cài uv thất bại" }
    Write-Host ">> Tạo .venv-mm3d (Python 3.10, uv tự tải Python nếu máy chưa có)"
    & $base -m uv venv .venv-mm3d --python 3.10 --seed
    if ($LASTEXITCODE -ne 0) { throw "Tạo môi trường thất bại" }
}
$PY = (Resolve-Path ".venv-mm3d\Scripts\python.exe").Path
# uv cài ngay trong .venv-mm3d: các lệnh sau không phụ thuộc Python nào đang mặc định trên máy
& $PY -m pip install --quiet uv
if ($LASTEXITCODE -ne 0) { throw "Cài uv vào .venv-mm3d thất bại" }
$MMCV = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html"

Write-Host ">> PyTorch 2.1.2 + CUDA 11.8 (~2.5 GB)"
& $PY -m uv pip install --python $PY torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw "Cài PyTorch thất bại" }

Write-Host ">> mmengine + mmcv 2.1.0 (bản dựng sẵn cho Windows, không build)"
& $PY -m uv pip install --python $PY "numpy<2" "setuptools<70" "mmengine>=0.10.3,<0.11" mmcv==2.1.0 --find-links $MMCV --only-binary mmcv
if ($LASTEXITCODE -ne 0) { throw "Cài mmcv thất bại (không có bản dựng sẵn cho tổ hợp này?)" }

Write-Host ">> mmdet 3.3.0 + mmdet3d 1.4.0 + nuscenes-devkit"
& $PY -m uv pip install --python $PY "numpy<2" "setuptools<70" mmdet==3.3.0 mmdet3d==1.4.0 nuscenes-devkit --find-links $MMCV --only-binary mmcv
if ($LASTEXITCODE -ne 0) { throw "Cài mmdet3d thất bại" }

Write-Host ">> Kiểm tra"
& $PY tools3d\run3d.py check
Write-Host ""
Write-Host "Xong. Chạy so sánh (khoảng 1 giờ cho 23 scene val):"
Write-Host "  .venv-mm3d\Scripts\python tools3d\run3d.py all --dataroot ..\v1.0-trainval"
