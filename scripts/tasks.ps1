# Các việc chạy trên terminal của máy Windows (PowerShell), chạy từ thư mục gốc repo P-130:
#
#   powershell -ExecutionPolicy Bypass -File scripts\tasks.ps1 <việc> [tuỳ chọn]
#
#   check       kiểm tra Python đang dùng có đủ thư viện (torch + CUDA, ultralytics, scipy, detector biển số...) và môi trường 3D
#   install     cài / cập nhật thư viện vào đúng Python đó (requirements-ml.txt gồm cả requirements.txt)
#   serve       chạy web (trang Dự án) bằng đúng Python đó: http://localhost:8000
#   test        ruff + pytest (giống CI)
#   demozip     đóng gói 1 scene nuScenes có đủ 6 camera + 10 sweep LiDAR thành zip để tải lên trang Dự án (demo 3D)
#   eval3d      chấm 3D trên các scene val có trên máy: 4 mô hình LiDAR (+ lật trục) rồi eval (tách dev / held-out)
#   label3d     tạo frame 3D cho UI "Workspace nhóm" bằng ensemble (chạy sau eval3d)
#   eval2d      chấm detector 2D trên dev / held-out (thêm -Weights để so trọng số đã fine-tune)
#   push        đẩy nhánh hiện tại lên GitHub
#   all         check -> test -> eval3d -> label3d -> eval2d
#
# Tuỳ chọn: -Dataroot ..\v1.0-trainval  -Scene scene-0035  -Weights weights\yoloe-26l-nuimages.pt  -NoTta  -Port 8000

param(
    [Parameter(Position = 0)]
    [ValidateSet("check", "install", "serve", "test", "demozip", "eval3d", "label3d", "eval2d", "push", "all")]
    [string]$Task = "check",
    [string]$Dataroot = "..\v1.0-trainval",
    [string]$Scene = "scene-0035",
    [string]$Weights = "",
    [switch]$NoTta,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
$Mm3d = ".venv-mm3d\Scripts\python.exe"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$argv) {
    Write-Host "> $exe $($argv -join ' ')" -ForegroundColor DarkGray
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Lệnh lỗi (mã $LASTEXITCODE): $exe $($argv -join ' ')" }
}

function Check {
    Step "Python dùng cho web và model 2D"
    $py = (Get-Command python).Source
    Write-Host "python = $py"
    $code = @"
import importlib.util, sys
mods = {'torch': 'torch', 'ultralytics': 'ultralytics', 'scipy': 'scipy', 'cv2': 'opencv-python-headless',
        'multipart': 'python-multipart', 'langgraph': 'langgraph', 'open_image_models': 'open-image-models'}
missing = [pip for m, pip in mods.items() if importlib.util.find_spec(m) is None]
print('missing:', ', '.join(missing) if missing else 'none')
if importlib.util.find_spec('torch'):
    import torch
    print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
if importlib.util.find_spec('ultralytics'):
    import ultralytics; print('ultralytics', ultralytics.__version__)
sys.exit(1 if missing else 0)
"@
    & python -c $code
    if ($LASTEXITCODE -ne 0) { Write-Host "-> chạy: scripts\tasks.ps1 install" -ForegroundColor Yellow }
    $uv = Get-Command uvicorn -ErrorAction SilentlyContinue
    if ($uv -and -not ($uv.Source.StartsWith((Split-Path $py)))) {
        Write-Host "Lưu ý: lệnh 'uvicorn' ($($uv.Source)) thuộc Python khác; luôn chạy web bằng: scripts\tasks.ps1 serve" -ForegroundColor Yellow
    }
    Step "Môi trường 3D (.venv-mm3d)"
    if (Test-Path $Mm3d) { Run $Mm3d @("tools3d\run3d.py", "check") }
    else { Write-Host "Chưa có .venv-mm3d: powershell -ExecutionPolicy Bypass -File tools3d\setup.ps1" -ForegroundColor Yellow }
}

function Install {
    Step "Cài thư viện vào $((Get-Command python).Source)"
    $hasCuda = (& python -c "import torch; print(torch.cuda.is_available())" 2>$null) -eq "True"
    if (-not $hasCuda) { Run "python" @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126") }
    Run "python" @("-m", "pip", "install", "-r", "requirements-ml.txt")
    Run "python" @("-m", "pip", "install", "git+https://github.com/ultralytics/CLIP.git")
}

function Serve {
    Step "Web: http://localhost:$Port (Ctrl+C để dừng)"
    Run "python" @("-m", "uvicorn", "src.main:app", "--port", "$Port")
}

function Test {
    Step "ruff + pytest"
    Run "python" @("-m", "ruff", "check", "src", "tests")
    Run "python" @("-m", "pytest", "-q")
}

function DemoZip {
    $out = "..\demo_$Scene.zip"
    Step "Đóng gói $Scene (6 camera + 10 sweep LiDAR) -> $out"
    Run "python" @("scripts\pack_nuscenes_subset.py", "--dataroot", $Dataroot, "--version", "v1.0-trainval", "--scenes", $Scene,
        "--all-cameras", "--lidar-sweeps", "10", "--with-lidar", "--workspace", "none", "--out", $out)
    Write-Host "Xong: tải $out lên trang Dự án (Loại dữ liệu: Tự nhận dạng)."
}

function Eval3d {
    if (-not (Test-Path $Mm3d)) { throw "Chưa có .venv-mm3d (tools3d\setup.ps1)" }
    Step "Suy luận 4 mô hình LiDAR trên scene val có trên máy$(if (-not $NoTta) { ' (+ lật trục, lâu hơn ~4 lần)' })"
    $argv = @("tools3d\run3d.py", "run", "--dataroot", $Dataroot, "-m", "pointpillars", "ssn", "centerpoint_pillar", "centerpoint_voxel")
    if (-not $NoTta) { $argv += "--tta" }
    Run $Mm3d $argv
    Step "Chấm mAP / NDS (dev = 3 scene demo, held-out = phần còn lại)"
    Run $Mm3d @("tools3d\run3d.py", "eval", "--dataroot", $Dataroot)
    Write-Host "Mỗi dòng 'dev ... heldout ...' ở trên là mAP/NDS; gửi các dòng đó (và eval\results\det3d\summary.json) cho nhóm." -ForegroundColor Green
}

function Label3d {
    Step "Tạo frame 3D cho UI bằng ensemble"
    Run "python" @("-m", "src.cli", "label3d", "--model", "ensemble")
    Run "python" @("-m", "src.cli", "evaluate3d", "--model", "ensemble")
}

function Eval2d {
    $w = @("yoloe-26l-seg.pt")
    if ($Weights) { $w += $Weights }
    Step "Chấm 2D trên CAM_FRONT: $($w -join ', ')"
    Run "python" (@("tools2d\eval2d.py", "--dataroot", $Dataroot, "--weights") + $w)
}

function Push {
    Step "git push"
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    git status --short
    Run "git" @("push", "-u", "origin", $branch)
}

switch ($Task) {
    "check" { Check }
    "install" { Install; Check }
    "serve" { Serve }
    "test" { Test }
    "demozip" { DemoZip }
    "eval3d" { Eval3d }
    "label3d" { Label3d }
    "eval2d" { Eval2d }
    "push" { Push }
    "all" { Check; Test; Eval3d; Label3d; Eval2d }
}
