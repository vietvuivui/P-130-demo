# tools2d — fine-tune YOLOE-26L trên nuImages

YOLOE-26L chạy zero-shot (prompt chữ) đạt mAP50 ≈ 0.40 trên CAM_FRONT của 3 scene dev. Các lớp yếu là barrier,
construction_vehicle, pedestrian, bicycle và motorcycle (AP 0.00–0.29). Đổi prompt, thêm prompt gây nhiễu hay lật ảnh
khi suy luận không cải thiện được đáng kể (xem `eval/results/det2d_variants.json`). Muốn tăng thật sự thì phải
**fine-tune**: dạy mô hình trên ảnh gán tay cùng miền dữ liệu.

**nuImages** là lựa chọn phù hợp:

- khoảng 93k ảnh của cùng đội xe và cùng thành phố với nuScenes (Boston, Singapore);
- box 2D gán tay, sát vật;
- cùng bộ tên lớp với nuScenes;
- không phải bộ dùng để chấm cuối.

## Quy trình chống overfit

| Dữ liệu | Dùng để |
|---|---|
| nuImages train (tự bỏ mọi ảnh chụp cùng xe, cùng ngày với một scene val của nuScenes, `tools2d/nuscenes_val_logs.json`) | học |
| nuImages val | chọn checkpoint (best.pt), dừng sớm |
| nuScenes val — 3 scene dev (scene-0035/0097/0101) | chỉ để xem, đã nhìn nhiều khi làm UI |
| nuScenes val — các scene held-out còn lại | **con số quyết định**: chỉ dùng trọng số mới khi held-out tốt hơn |

Siêu tham số lấy theo hướng dẫn của Ultralytics cho YOLOE (AdamW, lr 1e-3 cho linear probe, weight decay 0.025).
Không dò siêu tham số trên nuScenes.

## Cách nhanh: một script trên máy GPU (Windows)

```powershell
git clone -b kien https://github.com/AI20K-Build-Phase-Cohort-4/P-130.git; cd P-130
# đặt nuimages-v1.0-all-metadata.tgz + nuimages-v1.0-all-samples.tgz vào D:\nuimages rồi:
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 setup
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 data
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 convert
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 smoke      # vài phút: bắt lỗi môi trường trước
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 baseline
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 lp
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 full
powershell -ExecutionPolicy Bypass -File tools2d\train_pc.ps1 package
```

- Máy tắt hoặc Windows khởi động lại giữa chừng: `train_pc.ps1 resume -Run lp-1280` (hoặc `full-1280`) chạy tiếp từ
  `last.pt`, giữ nguyên dữ liệu, lr, số epoch và các lớp đóng băng.
- Chống rò rỉ không cần nuScenes trên máy GPU: danh sách 15 log (xe, ngày) chứa 150 scene val nằm sẵn trong
  `tools2d/nuscenes_val_logs.json`.
- `-Batch -1` (mặc định) để Ultralytics tự chọn batch theo VRAM. Lỗi hết bộ nhớ thì thêm `-Batch 8` (hoặc 4).
- Parsec không cần cho việc huấn luyện. Lỗi 6023 kèm 11002 là do mạng dùng NAT nhiều tầng (CGNAT hoặc hai router);
  11010 là router tắt UPnP. Chuyển file thì dùng croc / Drive ở bảng dưới; điều khiển máy từ xa thì Chrome Remote Desktop
  hoặc Tailscale + Remote Desktop chạy được qua CGNAT.

## 1. Chuẩn bị máy huấn luyện (vd. RTX 3090, không chung mạng LAN với máy chạy web)

**Code:** lấy từ GitHub, không cần chép qua lại. Repo private nên cần đăng nhập, bằng `gh auth login` hoặc
Personal Access Token:

```powershell
git clone -b kien https://github.com/AI20K-Build-Phase-Cohort-4/P-130.git
cd P-130
py -3.11 -m venv .venv; .\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-ml.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Dữ liệu:** tải thẳng về máy huấn luyện, không cần đi qua máy của bạn. Đăng ký tài khoản miễn phí ở
nuscenes.org → Download → nuImages, rồi tải:

- `nuimages-v1.0-all-metadata.tgz`: bảng json;
- `nuimages-v1.0-all-samples.tgz`: ảnh keyframe, khoảng 15 GB. **Không cần** file sweeps.
- Không cần tải nuScenes lên máy huấn luyện: danh sách log val để chống rò rỉ đã nằm sẵn trong repo.

Giải nén vào cùng một thư mục, ví dụ `D:\nuimages\` gồm `v1.0-train\`, `v1.0-val\`, `samples\`.

### Chuyển file giữa hai máy không chung LAN

Dùng khi máy huấn luyện không tự tải được, hoặc để mang `best.pt` (~100 MB) về máy chạy web.

| Cách | Khi nào | Lệnh |
|---|---|---|
| **croc** (P2P qua internet, mã hoá, nối lại được khi đứt) | tiện nhất cho 1 file bất kỳ cỡ nào | máy gửi: `winget install schollz.croc` rồi `croc send best.pt`, đọc mã (vd. `7-alpha-beta-gamma`); máy nhận: `croc 7-alpha-beta-gamma` |
| Google Drive / OneDrive | file vài trăm MB, hoặc muốn giữ lại | tải lên, chia sẻ link; file lớn dùng `rclone copy gdrive:nuimages D:\nuimages -P` (có tiếp tục khi đứt) |
| **Tailscale** (mạng riêng ảo, miễn phí) | chép nhiều lần, hoặc muốn dùng như chung LAN | cài trên cả hai máy, đăng nhập cùng tài khoản; máy kia có IP `100.x.y.z`, dùng `scp` / chia sẻ thư mục Windows như trong LAN |
| GitHub Release | chia `best.pt` cho cả nhóm | `gh release create yoloe-ft-v1 weights\yoloe-26l-nuimages.pt` (giới hạn 2 GB mỗi file) |

File rất lớn thì chia nhỏ trước khi gửi: `7z a -v4g nuimages.7z D:\nuimages`, gửi các phần `.001`, `.002`…

Luôn kiểm tra toàn vẹn ở hai đầu và so hai chuỗi:

- Windows: `certutil -hashfile best.pt SHA256`
- Linux: `sha256sum best.pt`

## 2. Đổi nuImages sang dataset YOLO

```powershell
python tools2d/nuimages_to_yolo.py --dataroot D:\nuimages --out D:\nuimages_yolo
```

- Ảnh được hardlink, nên không tốn thêm dung lượng ổ khi cùng ổ đĩa.
- `stats.json` ghi số ảnh, số box từng lớp và số ảnh bị loại vì trùng log.
- Thử nhanh trước bằng `--limit 200`.

## 3. Huấn luyện

```powershell
python tools2d/finetune_yoloe.py baseline --data D:\nuimages_yolo\data.yaml   # mốc zero-shot trên nuImages val
python tools2d/finetune_yoloe.py lp       --data D:\nuimages_yolo\data.yaml   # linear probe, 10 epoch
python tools2d/finetune_yoloe.py full     --data D:\nuimages_yolo\data.yaml --imgsz 960 --batch 8   # tuỳ chọn
```

**lp (linear probe)**

- Chỉ học lớp cuối của nhánh phân lớp. Lớp này được khởi tạo từ text embedding của tên lớp, nên bắt đầu đúng bằng
  mô hình zero-shot.
- Nhanh, ít VRAM, rất khó overfit: nên chạy trước.
- Ước lượng trên RTX 3090 ở 1280 px: khoảng 20–30 phút mỗi epoch.

**full**

- Học toàn bộ mạng với lr nhỏ (2e-4, cosine), tắt mosaic ở 10 epoch cuối, dừng sớm sau 8 epoch không tiến bộ.
- Lâu hơn nhiều (khoảng 30 phút mỗi epoch ở 960 px).
- Chỉ chạy khi lp đã hơn baseline.

**Hết VRAM:** giảm `--batch` hoặc `--imgsz 960`. **Thử cả luồng trong vài phút:** `--fraction 0.02 --epochs 1`.

Mỗi lần chạy lưu ở `runs/yoloe_ft/<stage>-<imgsz>/`. Checkpoint tốt nhất theo nuImages val là `weights/best.pt`,
còn `results.csv` là đường mAP theo từng epoch. Nếu mAP val đi xuống trong khi loss train vẫn giảm, đó là overfit:
best.pt đã giữ epoch tốt nhất.

## 4. Chấm trên held-out và dùng trọng số mới

Mang `best.pt` về máy chạy web (xem bảng ở mục 1), đặt ở `weights\yoloe-26l-nuimages.pt`. Đặt tên riêng như vậy vì
tên file nằm trong khoá cache detection. Sau đó chấm:

```powershell
python tools2d/eval2d.py --dataroot ..\v1.0-trainval --weights yoloe-26l-seg.pt weights\yoloe-26l-nuimages.pt
```

Script in mAP50, P, R, F1 và AP từng lớp cho dev và held-out, rồi ghi vào `eval/results/det2d_finetune.json`.
Khi đọc số cần nhớ: GT 2D của nuScenes là hộp bao của box 3D chiếu xuống ảnh, rộng hơn box sát vật mà nuImages dạy.
Vì vậy xem cả recall và AP từng lớp, không chỉ mAP50.

Nếu held-out tốt hơn, sửa `configs/autolabel.yaml`:

```yaml
detection:
  yoloe:
    weights: weights/yoloe-26l-nuimages.pt
```

- Detector tự nhận ra trọng số tập lớp đóng: tên lớp trùng tên lớp của config, nên bỏ qua prompt chữ.
- Cache detection cũ vẫn giữ, vì khoá theo tên file trọng số.
- Chạy `python -m src.cli run --overwrite` để gán nhãn lại.
- Khi đổi trọng số, chạy lại ngưỡng `min_score` bằng `eval/precision_tuning.ipynb`.

Lưu ý: fine-tune xong mô hình chỉ còn 10 lớp nuScenes. Muốn thêm lớp mới (open-vocab) thì quay về trọng số gốc.
