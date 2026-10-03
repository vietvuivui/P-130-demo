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
#   evalprop3d  đo lan truyền box 3D đã duyệt trên các scene val có dự đoán 3D (dùng dự đoán của eval3d, không cần GPU)
#   profile     đo một frame tốn thời gian ở bước nào (detect, LiDAR, QA, làm mờ ảnh, nạp model), không dùng cache: -Limit 10
#   evaltemporal  so sánh trước / sau optical flow (lan truyền nhãn, QA temporal, tính lại score theo sweep) trên
#               dev (3 scene demo) và held-out (-Scenes, mặc định 20 scene val có đủ dữ liệu); detect chạy GPU, có cache
#   rescore     bật / tắt tính lại score keyframe theo sweep trong configs/autolabel.yaml: -Mode off | mean | linked
#               (trên web: tab ⚙ Cài đặt làm được việc này và evaltemporal cho từng workspace / dự án, không cần terminal)
#   trackeval   HOTA / MOTA / IDF1 (chuẩn TrackEval) của nhãn lan truyền 2D + box 3D trong workspace (-Workspace),
#               ghi MOTChallenge và chạy TrackEval chính thức nếu đã cài
#   dam4sam     lan truyền box/mask bằng DAM4SAM (SAM 2.1, GPU) từ keyframe đã duyệt và so với tracker hiện tại
#               (-Scenes; cần clone repo DAM4SAM + checkpoint, hướng dẫn ở đầu tools2d/dam4sam.py)
#   improve     đo 4 cải tiến (10/2026) bằng số: detector ở 1920 / lưới ô cho vật nhỏ; không hạ điểm vật ngoài tầm LiDAR
#               khi gộp 3D; giữ box detector thấy mờ khi sweep hai bên thấy; mang nhãn frame trước bằng tracker.
#               Kết quả: eval\results\improve\*.json (+ bảng in ra). GPU ~10 phút cho detector, còn lại CPU ~20 phút
#   trackall    MỘT LỆNH cho mọi phép đo tracking trên dev (3 scene, hoặc -Scenes a,b): tự detect ảnh 12 Hz thiếu, đo tăng
#               tốc DAM4SAM, so 5 cấu hình, in bảng + khuyến nghị vào eval\results\trackall\dam4sam.md
#   push        đẩy nhánh hiện tại lên GitHub
#   all         check -> test -> eval3d -> label3d -> eval2d
#
# Tuỳ chọn: -Dataroot ..\v1.0-trainval  -Scene scene-0035  -Weights weights\yoloe-26l-nuimages.pt  -NoTta  -Port 8000
#           -Scenes scene-0003,scene-0016 (dấu phẩy; held-out cho evaltemporal)  -Mode mean (cho rescore)

param(
    [Parameter(Position = 0)]
    [ValidateSet("check", "install", "serve", "test", "demozip", "eval3d", "label3d", "eval2d", "evaltemporal", "rescore", "profile", "evalprop3d",
        "push", "trackeval", "dam4sam", "trackall", "improve", "all")]
    [string]$Task = "check",
    [string]$Dataroot = "..\v1.0-trainval",
    [string]$Scene = "scene-0035",
    [string]$Weights = "",
    [switch]$NoTta,
    [int]$Port = 8000,
    [string[]]$Scenes = @(),
    [string]$Model = "sam21pp-L",   # DAM4SAM: sam21pp-L | -B | -S | -T (nhỏ hơn = nhanh hơn)
    [int]$Stride = 1,               # DAM4SAM: chỉ xử lý mỗi ảnh thứ k giữa hai keyframe (3 nhanh gấp ~3, 6 = chỉ keyframe)
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

function EvalProp3d {
    Step "Lan truyền 3D: chỉ bù chuyển động xe / + vận tốc / + ngưỡng chặt (dev 3 scene, held-out phần còn lại)"
    Run "python" @("tools3d\eval_propagation3d.py", "--dataroot", $Dataroot)
}

function TrackEval {
    Step "HOTA / MOTA / IDF1 của nhãn 2D (track_id) và box 3D so với nhãn gốc"
    $saved = $env:WORKSPACE_DIR
    try {
        if ($Workspace) { $env:WORKSPACE_DIR = $Workspace }
        Run "python" @("-m", "src.cli", "trackeval", "--export-mot")
    } finally { $env:WORKSPACE_DIR = $saved }
}

function Dam4Sam {
    Step "DAM4SAM (SAM 2.1) lan truyền từ keyframe đã duyệt, so với tracker flow + BoT-SORT"
    $args_ = @("tools2d\dam4sam.py", "--dataroot", $Dataroot)
    if ($Workspace) { $args_ += @("--workspace", $Workspace) }
    if ($Scenes.Count) { $args_ += @("--scenes") + $Scenes }
    Run "python" $args_
}

function TrackAll {
    # Một lệnh cho mọi phép đo tracking trên dev: tự detect ảnh 12 Hz còn thiếu, đo tăng tốc DAM4SAM, chạy 5 cấu hình
    # (flow / DAM4SAM x ByteTrack / BoT-SORT / không ghép), in bảng HOTA / MOTA / IDF1 và khuyến nghị
    $dev = if ($Scenes.Count) { $Scenes } else { @("scene-0035", "scene-0097", "scene-0101") }
    $ws = if ($Workspace) { $Workspace } else { "data\eval_temporal\ws_dev" }
    Step "So sánh tracker trên $($dev -join ', ') ($ws)"
    # Biến thể nhanh (-Model sam21pp-T, -Stride 3 ...) ghi vào thư mục riêng để không đè kết quả của cấu hình gốc
    $out = "eval\results\trackall"
    $extra = @("--model", $Model, "--stride", $Stride)
    if ($Model -ne "sam21pp-L" -or $Stride -ne 1) { $out = "${out}_$($Model.Split('-')[-1])_k$Stride" } else { $extra += "--bench" }
    Run "python" (@("tools2d\dam4sam.py", "--dataroot", $Dataroot, "--workspace", $ws, "--out", $out) + $extra + @("--scenes") + $dev)
}

function Improve {
    $out = "eval\results\improve"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $w = "weights\yoloe-26l-nuimages-lp-1280.pt"
    Step "1/4 Detector: 1280 (mặc định) / 1920 / 1280 + lưới ô 2x2 — mAP50, recall theo cỡ vật (GPU)"
    Run "python" @("tools2d\eval2d.py", "--dataroot", $Dataroot, "--weights", $w, "--out", "$out\det2d.json")
    Run "python" @("tools2d\eval2d.py", "--dataroot", $Dataroot, "--weights", $w, "--imgsz", "1920", "--out", "$out\det2d.json")
    Run "python" @("tools2d\eval2d.py", "--dataroot", $Dataroot, "--weights", $w, "--tiles", "2", "--out", "$out\det2d.json")
    Step "2/4 Gộp box 3D vào nhãn 2D: hạ điểm box chỉ camera thấy (0.5) so với không hạ khi box không có điểm LiDAR (CPU)"
    Run "python" @("tools2d\eval_lidar2d.py", "--dataroot", $Dataroot, "--out", "$out\lidar2d.json")
    foreach ($split in @(@("dev", "data\eval_temporal\ws_dev"), @("heldout", "data\eval_temporal\ws_heldout"))) {
        $name, $ws = $split[0], $split[1]
        Step "3-4/4 QA temporal trên $name ($ws): noweak (trước 02/10) / weak (giữ box thấy mờ) / carry (+ mang nhãn frame trước) (CPU)"
        Run "python" @("tools2d\eval_temporal.py", "--dataroot", $Dataroot, "--workspace", $ws, "--out", "$out\temporal_$name",
            "--variants", "noweak", "weak", "carry", "carry-only", "--no-propagation")
    }
    Write-Host "Xong. Gửi: $out\det2d.json, lidar2d.json, temporal_dev\temporal_eval.json, temporal_heldout\temporal_eval.json (hoặc chép bảng in ra)." -ForegroundColor Green
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
    "evalprop3d" { EvalProp3d }
    "trackeval" { TrackEval }
    "dam4sam" { Dam4Sam }
    "trackall" { TrackAll }
    "improve" { Improve }
    "push" { Push }
    "all" { Check; Test; Eval3d; Label3d; Eval2d }
}
