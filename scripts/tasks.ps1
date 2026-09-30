# Các việc chạy trên terminal của máy Windows (PowerShell), chạy từ thư mục gốc repo P-130:
#
#   powershell -ExecutionPolicy Bypass -File scripts\tasks.ps1 <việc> [tuỳ chọn]
#
#   check       kiểm tra Python đang dùng có đủ thư viện (torch + CUDA, ultralytics, scipy, detector biển số...) và môi trường 3D
#   install     cài / cập nhật thư viện vào đúng Python đó (requirements-ml.txt gồm cả requirements.txt)
#   serve       chạy web (trang Dự án) bằng đúng Python đó: http://localhost:8000
#               -Workspace <thư mục> [-Dataroot <nuScenes>]: mở một workspace khác (vd. để demo), không sửa .env
#   test        ruff + pytest (giống CI)
#   demozip     đóng gói 1 scene nuScenes có đủ 6 camera + 10 sweep LiDAR thành zip để tải lên trang Dự án (demo 3D)
#   eval3d      chấm 3D trên các scene val có trên máy: 4 mô hình LiDAR (+ lật trục) rồi eval (tách dev / held-out)
#   label3d     tạo frame 3D cho UI "Workspace nhóm" bằng ensemble (chạy sau eval3d)
#   eval2d      chấm detector 2D trên dev / held-out (thêm -Weights để so trọng số đã fine-tune)
#   profile     đo một frame tốn thời gian ở bước nào (detect, LiDAR, QA, làm mờ ảnh, nạp model), không dùng cache: -Limit 10
#   evaltemporal  so sánh trước / sau optical flow (lan truyền nhãn, QA temporal, tính lại score theo sweep) trên
#               dev (3 scene demo) và held-out (-Scenes, mặc định 20 scene val có đủ dữ liệu); detect chạy GPU, có cache
#   rescore     bật / tắt tính lại score keyframe theo sweep trong configs/autolabel.yaml: -Mode off | mean | linked
#               (trên web: tab ⚙ Cài đặt làm được việc này và evaltemporal cho từng workspace / dự án, không cần terminal)
#   push        đẩy nhánh hiện tại lên GitHub
#   all         check -> test -> eval3d -> label3d -> eval2d
#
# Tuỳ chọn: -Dataroot ..\v1.0-trainval  -Scene scene-0035  -Weights weights\yoloe-26l-nuimages.pt  -NoTta  -Port 8000
#           -Scenes scene-0003 scene-0016 (held-out cho evaltemporal)  -Mode mean (cho rescore)

param(
    [Parameter(Position = 0)]
    [ValidateSet("check", "install", "serve", "test", "demozip", "eval3d", "label3d", "eval2d", "evaltemporal", "rescore", "profile",
        "push", "all")]
    [string]$Task = "check",
    [string]$Dataroot = "..\v1.0-trainval",
    [string]$Scene = "scene-0035",
    [string]$Weights = "",
    [switch]$NoTta,
    [int]$Port = 8000,
    [string[]]$Scenes = @(),
    [string]$Workspace = "",
    [int]$Limit = 10,
    [ValidateSet("off", "mean", "linked")]
    [string]$Mode = "off"
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
    if ($Workspace) {
        $env:WORKSPACE_DIR = $Workspace
        if ($PSBoundParameters.ContainsKey("Dataroot")) { $env:NUSCENES_DATAROOT = $Dataroot; $env:NUSCENES_VERSION = "v1.0-trainval" }
        Write-Host "Workspace: $Workspace$(if ($env:NUSCENES_DATAROOT) { " · nuScenes: $env:NUSCENES_DATAROOT" })" -ForegroundColor Green
    }
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

function EvalTemporal {
    $dev = @("scene-0035", "scene-0097", "scene-0101")
    $held = if ($Scenes.Count) { $Scenes } else {
        @("scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015", "scene-0016", "scene-0017", "scene-0018",
          "scene-0036", "scene-0038", "scene-0039", "scene-0092", "scene-0093", "scene-0094", "scene-0095", "scene-0096",
          "scene-0098", "scene-0099", "scene-0100", "scene-0102")
    }
    $saved = @{ NUSCENES_DATAROOT = $env:NUSCENES_DATAROOT; NUSCENES_VERSION = $env:NUSCENES_VERSION; WORKSPACE_DIR = $env:WORKSPACE_DIR }
    try {
        foreach ($split in @(@("dev", $dev), @("heldout", $held))) {
            $name, $list = $split[0], $split[1]
            $ws = "data\eval_temporal\ws_$name"
            $env:NUSCENES_DATAROOT = $Dataroot
            $env:NUSCENES_VERSION = "v1.0-trainval"
            $env:WORKSPACE_DIR = $ws
            Step "Auto-label $name ($($list.Count) scene) vào $ws (detect keyframe + sweep t-2..t+2, có cache)"
            Run "python" (@("-m", "src.cli", "run", "--scenes") + $list)
            Step "Trước / sau optical flow: $name"
            Run "python" @("tools2d\eval_temporal.py", "--dataroot", $Dataroot, "--workspace", $ws, "--out", "data\eval_temporal\result_$name")
        }
    }
    finally {
        foreach ($k in $saved.Keys) {
            if ($saved[$k]) { Set-Item "env:$k" $saved[$k] } else { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
        }
    }
    Write-Host "Kết quả đầy đủ: data\eval_temporal\result_dev\temporal_eval.json, result_heldout\temporal_eval.json" -ForegroundColor Green
}

function Rescore {
    $path = "configs\autolabel.yaml"
    $text = [IO.File]::ReadAllText((Resolve-Path $path))
    $flow = if ($Mode -eq "off") { "false" } else { "true" }
    $text = [regex]::Replace($text, '(?m)^    flow: (true|false)', "    flow: $flow")
    $text = [regex]::Replace($text, '(?m)^    rescore: "?(off|mean|linked)"?', "    rescore: ""$Mode""")
    [IO.File]::WriteAllText((Resolve-Path $path), $text, (New-Object System.Text.UTF8Encoding($false)))
    Step "qa.temporal.flow = $flow, qa.temporal.rescore = $Mode"
    Select-String -Path $path -Pattern '^    (flow|rescore):' | ForEach-Object { Write-Host $_.Line }
    Write-Host "Áp cho frame mới auto-label. Frame cũ chưa ai mở: python -m src.cli run --overwrite (dùng cache detection, nhanh)." -ForegroundColor Yellow
    Write-Host "Khởi động lại web (scripts\tasks.ps1 serve) để server đọc config mới." -ForegroundColor Yellow
}

function Profile {
    Step "Đo thời gian từng bước trên $Limit keyframe (workspace tạm, không dùng cache)"
    $argv = @("-m", "src.cli", "profile", "--limit", "$Limit")
    if ($Scenes.Count) { $argv += @("--scenes") + $Scenes }
    if ($PSBoundParameters.ContainsKey("Dataroot")) { $env:NUSCENES_DATAROOT = $Dataroot; $env:NUSCENES_VERSION = "v1.0-trainval" }
    Run "python" $argv
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
    "evaltemporal" { EvalTemporal }
    "rescore" { Rescore }
    "profile" { Profile }
    "push" { Push }
    "all" { Check; Test; Eval3d; Label3d; Eval2d }
}
