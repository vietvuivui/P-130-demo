# Fine-tune YOLOE-26L trên nuImages ở máy có GPU (Windows, vd. RTX 3090). Chạy từ thư mục gốc repo P-130:
#
#   powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 <bước> [tuỳ chọn]
#
#   setup      tạo môi trường .venv-train (Python 3.11, PyTorch CUDA, Ultralytics), kiểm tra GPU, tắt ngủ máy khi cắm điện
#   data       giải nén nuimages-v1.0-all-metadata.tgz + nuimages-v1.0-all-samples.tgz (đặt trong -Nuimages) nếu chưa giải nén
#   convert    nuImages -> dataset YOLO 10 lớp (-Out), tự bỏ ảnh cùng xe + cùng ngày với scene val nuScenes (chống rò rỉ)
#   smoke      chạy thử cả luồng trong vài phút (2% dữ liệu, 1 epoch) để bắt lỗi môi trường trước khi chạy thật
#   baseline   mAP zero-shot trên nuImages val (mốc so sánh)
#   lp         linear probe: chỉ học lớp cuối nhánh phân lớp (10 epoch)
#   full       fine-tune toàn bộ mạng (30 epoch, dừng sớm theo nuImages val)
#   resume     chạy tiếp một lần huấn luyện bị ngắt (máy tắt / Windows cập nhật): -Run lp-1280 hoặc full-1280
#   package    chép best.pt thành weights\yoloe-26l-nuimages-<run>.pt kèm SHA256 để mang về máy chạy web
#   all        setup -> data -> convert -> smoke -> baseline -> lp -> full -> package
#
# Tuỳ chọn: -Nuimages D:\nuimages  -Out D:\nuimages_yolo  -Imgsz 1280  -Batch -1 (tự chọn theo VRAM)  -Run lp-1280

param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "data", "convert", "smoke", "baseline", "lp", "full", "resume", "package", "all")]
    [string]$Step = "setup",
    [string]$Nuimages = "D:\nuimages",
    [string]$Out = "D:\nuimages_yolo",
    [int]$Imgsz = 1280,
    [int]$Batch = -1,
    [int]$Workers = 8,
    [string]$Run = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
$Py = ".venv-train\Scripts\python.exe"
$Data = Join-Path $Out "data.yaml"

function Say($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$argv) {
    Write-Host "> $exe $($argv -join ' ')" -ForegroundColor DarkGray
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Lệnh lỗi (mã $LASTEXITCODE): $exe $($argv -join ' ')" }
}
function NeedEnv { if (-not (Test-Path $Py)) { throw "Chưa có .venv-train: chạy bước setup trước" } }

function Setup {
    Say "GPU"
    Run "nvidia-smi" @("--query-gpu=name,memory.total,driver_version", "--format=csv")
    if (-not (Test-Path $Py)) {
        Say "Tạo .venv-train (Python 3.11)"
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) { Run "py" @("-3.11", "-m", "venv", ".venv-train") }
        else { Run "python" @("-m", "venv", ".venv-train") }
    }
    Say "Cài thư viện (PyTorch CUDA 12.6, Ultralytics)"
    Run $Py @("-m", "pip", "install", "--upgrade", "pip")
    Run $Py @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126")
    Run $Py @("-m", "pip", "install", "-r", "requirements.txt", "ultralytics>=8.4")
    Run $Py @("-m", "pip", "install", "git+https://github.com/ultralytics/CLIP.git")
    Run $Py @("-c", "import torch, ultralytics; assert torch.cuda.is_available(), 'torch khong thay GPU'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| ultralytics', ultralytics.__version__)")
    Say "Không cho máy ngủ khi cắm điện (huấn luyện chạy nhiều giờ)"
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Write-Host "Nên tạm dừng Windows Update vài ngày (Settings > Windows Update > Pause); nếu máy vẫn khởi động lại thì dùng bước resume."
}

function Data {
    Say "Giải nén nuImages vào $Nuimages"
    New-Item -ItemType Directory -Force -Path $Nuimages | Out-Null
    foreach ($pair in @(@("nuimages-v1.0-all-metadata.tgz", "v1.0-train"), @("nuimages-v1.0-all-samples.tgz", "samples"))) {
        $tgz = Join-Path $Nuimages $pair[0]
        if (Test-Path (Join-Path $Nuimages $pair[1])) { Write-Host "$($pair[1]) đã có, bỏ qua"; continue }
        if (-not (Test-Path $tgz)) { throw "Không thấy $tgz (tải từ nuscenes.org > Download > nuImages, hoặc chép từ máy khác)" }
        Run "tar" @("-xzf", $tgz, "-C", $Nuimages)
    }
    foreach ($d in @("v1.0-train", "v1.0-val", "samples")) {
        if (-not (Test-Path (Join-Path $Nuimages $d))) { throw "Thiếu $Nuimages\$d sau khi giải nén" }
    }
    Write-Host "OK: $Nuimages có v1.0-train, v1.0-val, samples. Có thể xoá 2 file .tgz để lấy lại ổ đĩa."
}

function Convert {
    NeedEnv
    Say "nuImages -> dataset YOLO ($Out)"
    Run $Py @("tools2d\nuimages_to_yolo.py", "--dataroot", $Nuimages, "--out", $Out)
}

function Smoke {
    NeedEnv
    Say "Chạy thử 1 epoch trên 2% dữ liệu (vài phút)"
    Run $Py @("tools2d\finetune_yoloe.py", "lp", "--data", $Data, "--imgsz", "640", "--batch", "8", "--epochs", "1",
        "--fraction", "0.02", "--workers", "$Workers", "--name", "smoke")
    Remove-Item -Recurse -Force "runs\yoloe_ft\smoke*" -ErrorAction SilentlyContinue
}

function Train([string]$stage) {
    NeedEnv
    Say "$stage ($Imgsz px, batch $Batch)"
    Run $Py @("tools2d\finetune_yoloe.py", $stage, "--data", $Data, "--imgsz", "$Imgsz", "--batch", "$Batch", "--workers", "$Workers")
}

function Baseline {
    NeedEnv
    Say "baseline zero-shot trên nuImages val"
    Run $Py @("tools2d\finetune_yoloe.py", "baseline", "--data", $Data, "--imgsz", "$Imgsz", "--batch", "16", "--workers", "$Workers")
}

function Resume {
    NeedEnv
    if (-not $Run) { throw "Cần -Run, vd. -Run lp-1280 (tên thư mục trong runs\yoloe_ft)" }
    $last = "runs\yoloe_ft\$Run\weights\last.pt"
    if (-not (Test-Path $last)) { throw "Không thấy $last" }
    Say "Chạy tiếp $Run"
    Run $Py @("tools2d\finetune_yoloe.py", "lp", "--resume", $last)
}

function Package {
    $runs = if ($Run) { @($Run) } else { Get-ChildItem "runs\yoloe_ft" -Directory | Where-Object { Test-Path "$($_.FullName)\weights\best.pt" } | ForEach-Object { $_.Name } }
    New-Item -ItemType Directory -Force -Path weights | Out-Null
    foreach ($r in $runs) {
        $src = "runs\yoloe_ft\$r\weights\best.pt"
        $dst = "weights\yoloe-26l-nuimages-$r.pt"
        Copy-Item $src $dst -Force
        $h = (Get-FileHash $dst -Algorithm SHA256).Hash
        Set-Content "$dst.sha256" "$h  $(Split-Path $dst -Leaf)"
        Write-Host "$dst  SHA256 $h"
        Copy-Item "runs\yoloe_ft\$r\results.csv" "weights\results-$r.csv" -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Mang file .pt (+ .sha256, results-*.csv) về máy chạy web: croc send weights\yoloe-26l-nuimages-*.pt  (xem tools2d\README.md)"
}

switch ($Step) {
    "setup" { Setup }
    "data" { Data }
    "convert" { Convert }
    "smoke" { Smoke }
    "baseline" { Baseline }
    "lp" { Train "lp" }
    "full" { Train "full" }
    "resume" { Resume }
    "package" { Package }
    "all" { Setup; Data; Convert; Smoke; Baseline; Train "lp"; Train "full"; Package }
}
