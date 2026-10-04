# Eval evidence — test case thủ công với output thực tế

Ngày chạy: 2026-10-01 (TC-01 → TC-11), 2026-10-02 (TC-12 → TC-14). Mỗi test case ghi: mục tiêu, bước làm trên UI, lời gọi API tương đương (UI gọi đúng API này),
kết quả mong đợi, **output thực tế** (chép nguyên từ lần chạy), và kết luận.

**Dữ liệu:** workspace `data/eval_temporal/ws_dev`, gồm 3 scene nuScenes (scene-0035, scene-0097, scene-0101) với 119
keyframe CAM_FRONT. Nhãn máy do YOLOE-26-L zero-shot + QA Agent sinh ra.

**Cách chạy:**

- **TC-01 → TC-07:** chạy trên bản sao của workspace, để không đụng dữ liệu thật.
- **Lệnh mở UI:** `scripts\tasks.ps1 serve -Workspace data\eval_temporal\ws_dev -Dataroot ..\v1.0-trainval`. Thao tác
  trên UI và gọi API cho ra cùng một kết quả.
- **TC-08 → TC-10:** dùng lệnh đánh giá trên tập test, so với nhãn gốc nuScenes.
- **TC-11:** một lỗi đã tìm ra khi kiểm thử thủ công.
- **TC-12, TC-13:** chạy trên workspace demo (`python -m src.demo`, clip dashcam 2560×1440 tải lên), máy không có GPU.
- **TC-14:** `scripts\tasks.ps1 trackall` trên RTX 4050.

| TC | Chức năng | Kết quả |
|---|---|---|
| TC-01 | Hàng đợi frame sắp theo rủi ro | ✅ Đạt |
| TC-02 | QA Agent giải thích vì sao box rủi ro | ✅ Đạt |
| TC-03 | Chặn approve khi còn box chưa duyệt | ✅ Đạt |
| TC-04 | Sửa nhãn: xoá, đổi lớp, thêm box, hoàn tác / làm lại, correction log | ✅ Đạt |
| TC-05 | Duyệt theo lô nhóm rủi ro thấp rồi approve frame | ✅ Đạt |
| TC-06 | Metrics cập nhật sau khi duyệt | ✅ Đạt |
| TC-07 | Xuất dataset COCO chỉ gồm frame đã approve | ✅ Đạt |
| TC-08 | Detector 2D fine-tune so với zero-shot trên tập test | ✅ Đạt (mAP50 0.312 → 0.589) |
| TC-09 | Lan truyền nhãn 2D qua các keyframe sau | ✅ Đạt (77% box ghi ra đúng vật) |
| TC-10 | Auto-label 3D trên tập test | ✅ Đạt (mAP 0.668 / NDS 0.713) |
| TC-11 | Kiểm tra temporal với vật ở gần đang chạy nhanh qua ảnh | ⚠️ Lỗi đã biết |
| TC-12 | Chọn vật: bấm điểm / kéo khung thô, SAM 2.1 ONNX trả mask + box | ✅ Đạt (bấm trúng một chi tiết thì chỉ ra chi tiết đó) |
| TC-13 | Nhiều người dùng: bắt đăng nhập, khoá frame đang mở | ✅ Đạt |
| TC-14 | So sánh tracker bằng TrackEval | ✅ Đạt (held-out: HOTA 0.563 mặc định, 0.573 với DAM4SAM + BoT-SORT) |

---

## TC-01 · Hàng đợi frame sắp theo rủi ro

- **Mục tiêu:** người duyệt mở frame khó nhất trước (review by exception).
- **Bước:** mở tab Review → ô sắp xếp chọn "Khó nhất trước".
- **API:** `GET /api/v1/frames?sort=risk`
- **Mong đợi:** 119 frame, `frame_risk` giảm dần, mỗi frame có số box theo mức low / medium / high.
- **Output thực tế:**

```text
200  n = 119, sắp giảm dần: True
[["scene-0101_023", 0.834, {"low": 18, "medium": 4, "high": 1}],
 ["scene-0101_022", 0.732, {"low": 16, "medium": 7, "high": 1}],
 ["scene-0101_016", 0.714, {"low": 16, "medium": 10, "high": 1}]]
```

- **Kết luận:** Đạt.

## TC-02 · QA Agent giải thích vì sao box rủi ro

- **Mục tiêu:** mỗi box có mức rủi ro và lý do, để người duyệt biết cần nhìn chỗ nào.
- **Bước:** mở `scene-0101_023` → bấm card đỏ đầu tiên (High).
- **API:** `GET /api/v1/frames/scene-0101_023`
- **Mong đợi:** box rủi ro cao nhất có issue code kèm lời giải thích; box chắc chắn rơi vào nhóm low.
- **Output thực tế** (5 box rủi ro cao nhất trong 23 box):

```text
#23 traffic_cone 0.3469  high    risk=0.834  [LOW_CONFIDENCE, NO_LIDAR_SUPPORT, FLICKER, ASPECT_RATIO_ABNORMAL]
#18 pedestrian   0.5293  medium  risk=0.425  [FLICKER]
#22 traffic_cone 0.363   medium  risk=0.423  [FLICKER]
#21 traffic_cone 0.4243  medium  risk=0.402  [FLICKER]
#13 pedestrian   0.6758  medium  risk=0.373  [FLICKER]

Lời giải thích của #23:
- LOW_CONFIDENCE         Score 0.35 < 0.35
- NO_LIDAR_SUPPORT       Chỉ 1 điểm LiDAR trong box cao 85px (cần ≥ 3)
- FLICKER                Chỉ xuất hiện ở 0/4 sweep lân cận
- ASPECT_RATIO_ABNORMAL  Tỉ lệ rộng/cao 0.18 ngoài [0.2, 1.4] của lớp traffic_cone
```

- **Kết luận:** Đạt. Box #23 đúng là cần sửa. Nhãn gốc ở đó là một cone bị cắt ở mép ảnh, `[1557, 450, 1600, 554]`;
  box máy quá hẹp nên IoU chỉ 0.29. Ở TC-04, người duyệt xoá box này để thử thao tác xoá; cách đúng hơn là kéo lại box
  (`EDIT_BOX`).

## TC-03 · Chặn approve khi còn box chưa duyệt

- **Bước:** mở `scene-0101_023` → bấm Approve ngay, chưa xem box nào.
- **API:** `POST /api/v1/frames/scene-0101_023/approve`
- **Mong đợi:** bị từ chối, báo còn bao nhiêu box chờ.
- **Output thực tế:**

```text
409 {"detail": {"code": "PENDING_OBJECTS",
               "message": "Còn 23 object chưa duyệt: 23, 18, 22, 21, 13, 20, 19, 10, 15, 17"}}
```

- **Kết luận:** Đạt.

## TC-04 · Sửa nhãn, hoàn tác / làm lại, correction log

- **Bước:**
  1. Xoá #23 (phím `Del`).
  2. Đổi lớp #18 sang barrier.
  3. Vẽ thêm một cone (`B`).
  4. `Ctrl+Z`, rồi `Ctrl+Y`.
  5. Mở tab Correction log.
- **API:** `POST /api/v1/frames/{id}/actions` (`DELETE`, `CHANGE_CLASS`, `ADD_BOX`), `POST /undo`, `POST /redo`,
  `GET /api/v1/corrections?frame_id=scene-0101_023`.
- **Mong đợi:** mỗi thao tác được lưu. Hoàn tác bỏ box vừa vẽ, làm lại thì box quay lại. Correction log ghi dự đoán gốc,
  QA và thao tác của người.
- **Output thực tế:**

```text
DELETE #23 → 200
CHANGE_CLASS #18 pedestrian → barrier (action CHANGE_CLASS) → 200
ADD_BOX → 200  [["h1", "traffic_cone", [1500.0, 640.0, 1560.0, 760.0], "human"]]
UNDO → 200  số box người vẽ: 0
REDO → 200  số box người vẽ: 1
Correction log: 5 dòng; dòng của #23:
{"object_id": "23", "prediction": {"bbox": [1575.0, 457.8, 1590.0, 542.8], "class": "traffic_cone",
 "score": 0.3469, "source": "model"}, "qa": {"risk": 0.834, "level": "high", "issues": ["LOW_CONFIDENCE",
 "NO_LIDAR_SUPPORT", "FLICKER", "ASPECT_RATIO_ABNORMAL"]}, "human_action": "DELETE", "reviewer": "annotator", ...}
```

- **Kết luận:** Đạt.

## TC-05 · Duyệt theo lô nhóm rủi ro thấp rồi approve

- **Bước:**
  1. Bấm "Duyệt nhóm low".
  2. Xem từng box medium còn lại → Giữ.
  3. Approve.
- **API:** `POST /approve-low-risk`, rồi `POST /actions` `KEEP` cho từng box còn lại, rồi `POST /approve`.
- **Mong đợi:** 18 box low được duyệt một lần. Chỉ còn các box medium chưa xem. Approve thành công.
- **Output thực tế:**

```text
approve-low-risk → 200  còn chờ: 3  [["22","traffic_cone","medium"], ["21","traffic_cone","medium"], ["13","pedestrian","medium"]]
approve → 200  status = approved, approved_by = annotator
```

- **Kết luận:** Đạt. Người duyệt chỉ phải xem 5 trong 23 box, vì 18 box còn lại được duyệt một lần theo lô.

## TC-06 · Metrics cập nhật sau khi duyệt

- **Bước:** mở tab Metrics & Export.
- **API:** `GET /api/v1/metrics`
- **Output thực tế:**

```text
frames: total 119, approved 1
objects_by_level: low 817, medium 352, high 8
m4_correction_rate: 0.087    (2 / 23 box máy phải sửa)
fix_rate_by_level: low 0/18, medium 1/4, high 1/1
flag_precision 0.40, flag_recall 1.00   (mọi box phải sửa đều đã bị gắn cờ)
```

- **Kết luận:** Đạt. Cả hai box phải sửa đều nằm trong nhóm bị gắn cờ, còn 18 box low không có box nào phải sửa.

## TC-07 · Xuất dataset COCO

- **Bước:** tab Metrics & Export → Xuất dataset.
- **API:** `POST /api/v1/export`
- **Mong đợi:** chỉ frame đã approve được xuất. Box đã xoá không có, box đổi lớp mang lớp mới.
- **Output thực tế:**

```text
200 {"n_frames": 1, "n_objects": 23, "files": ["coco.json", "labels.jsonl", "corrections.jsonl", "manifest.json"]}
coco.json: 1 ảnh, 23 annotation (23 máy − 1 xoá + 1 người vẽ), 10 lớp nuScenes
annotation đầu: {"category_id": 10 (barrier), "object_id": "18", "source": "model", "model_score": 0.5293,
                 "risk": 0.425, "issues": ["FLICKER"], "human_action": "CHANGE_CLASS", "reviewer": "annotator"}
```

- **Kết luận:** Đạt.

## TC-08 · Detector 2D fine-tune so với zero-shot

- **Lệnh:** `scripts\tasks.ps1 eval2d -Weights weights\yoloe-26l-nuimages-full-1280.pt` (và `…-lp-1280.pt` cho bản linear
  probe), chạy trên GPU laptop.
- **Dữ liệu:** 957 keyframe CAM_FRONT của 24 scene test, so với nhãn gốc nuScenes.
- **Mong đợi:** mô hình fine-tune tốt hơn zero-shot trên test.
- **Output thực tế** (`eval/results/det2d_finetune.json`):

```text
yoloe-26l-seg.pt               heldout  mAP50 0.312  P 0.499  R 0.523  F1 0.511  (957 ảnh)
yoloe-26l-nuimages-lp-1280.pt  heldout  mAP50 0.366  P 0.484  R 0.581  F1 0.528  (957 ảnh)
   AP: barrier 0.13→0.33  bicycle 0.12→0.23  traffic_cone 0.39→0.50  pedestrian 0.23→0.29  bus 0.81→0.78  truck 0.52→0.47
yoloe-26l-nuimages-full-1280.pt heldout mAP50 0.589  P 0.487  R 0.806  F1 0.607  (957 ảnh)   [chấm 2026-10-04]
   AP: barrier 0.68  bicycle 0.63  bus 0.89  car 0.85  construction_vehicle 0.11  motorcycle 0.54  pedestrian 0.56  traffic_cone 0.80  trailer 0.22  truck 0.61
   recall theo cỡ vật (cao px): 0-32 184/268  32-64 1575/1906  64-128 2246/2570  128- 1575/1989
```

- **Kết luận:** Đạt. Bản fine-tune toàn mạng (`full-1280`) là detector mặc định; linear probe và bản gốc chọn được ở tab ⚙
  Cài đặt (Fine-tune mini / Original).

## TC-09 · Lan truyền nhãn 2D

- **Bước trên UI:** chế độ Video → duyệt keyframe đầu → Approve, với ô "Lan truyền khi approve" đang bật → mở các
  keyframe sau, box có tag ↦.
- **Đo tự động:** nhãn gốc ở keyframe 0, 5, 10… đóng vai nhãn người, lan truyền tối đa 10 keyframe. Chạy trên 20 scene
  test (154 lần lan truyền).
- **Output thực tế** (`eval/results/tracker_test20.json`, cấu hình mặc định: optical flow + ByteTrack):

```text
box ghi ra 4128: đúng vật 3192 (77.3%), box sai 860, đổi ID 76, vật bị thiếu 1298
```

- **Kết luận:** Đạt. Người duyệt chỉ phải sửa phần box sai. Bảng so sánh các tracker: `eval/results/bang-so-sanh.md`.

## TC-10 · Auto-label 3D

- **Lệnh:** `scripts\tasks.ps1 eval3d`, dự đoán 4 mô hình LiDAR rồi gộp và tinh chỉnh theo track.
- **Dữ liệu:** 24 scene test, 957 keyframe, so với nhãn gốc.
- **Output thực tế** (`eval/results/det3d_heldout.md`):

```text
CenterPoint voxel           mAP 0.578  NDS 0.655
Gộp 4 LiDAR + track         mAP 0.668  NDS 0.713   (mặc định)
AP: car 0.886  pedestrian 0.900  bus 0.803  traffic_cone 0.802  trailer 0.620  truck 0.587
```

- **Kết luận:** Đạt.

## TC-11 · Lỗi đã biết: kiểm tra temporal với vật ở gần

- **Bước:** dự án `nusc3d_scene-0101` → frame `scene-0101_023` → bấm cone #2 (cách xe 8 m, mép phải ảnh) → xem dải
  Temporal consistency.
- **Mong đợi:** cone có mặt ở cả 4 sweep t−2…t+2.
- **Output thực tế:** UI báo t−1 **missing**. Detection đã cache của từng ảnh như sau. Tâm theo trục x của box ở keyframe
  là 1455 px.

```text
keyframe  cone #2  [1419, 622, 1491, 747]
t−2 (−200 ms)  ghép với [1436, 630, 1501, 755]   IoU 0.60 → "detected"
t−1 (−100 ms)  có cone [1377, 614, 1442, 730] score 0.88, nhưng IoU 0.18 < 0.3 → "missing"
t+1 ( +50 ms)  ghép với [1445, 627, 1520, 760]   → "detected"
t+2 (+150 ms)  ghép với [1435, 533, 1525, 750] score 0.72; cone [1500, 641, 1590, 788] score 0.89 không được chọn
```

- **Phân tích:**
  - Cone ở gần dịch khoảng 0.5 px/ms sang phải (+26 px sau 50 ms). Ở t−1, cone thật nằm ở [1377…1442]: detector có thấy
    (0.88), nhưng box dịch xa hơn nửa bề rộng nên IoU chỉ còn 0.18 và bị tính là thiếu.
  - Ở t−2 và t+2, phép so IoU lại ghép nhầm sang **cone bên cạnh** trong hàng cone, vì hàng cone đều nhau.
  - Nguyên nhân: kiểm tra temporal so box sweep với box keyframe bằng IoU trực tiếp, không bù chuyển động
    (`qa.temporal.flow: false`).
- **Ảnh hưởng:** nhỏ. Cone #2 vẫn ở nhóm low (risk 0.06, có mặt 3/4 sweep, LiDAR 27 điểm), nên không làm sai quyết định
  duyệt. Lỗi này chỉ làm cờ FLICKER kém chính xác với vật ở gần chạy nhanh qua ảnh.
- **Hướng sửa:**
  - Bật optical flow cho QA temporal (đã có, IoU trung bình ở t±2 tăng từ 0.58 lên 0.83). Nhưng trên 20 scene test,
    lỗi lọt qua duyệt theo lô lại tăng (1632 → 2082), nên hiện để tắt.
  - Hoặc ghép theo chuỗi t → t±1 → t±2 với vận tốc không đổi. Việc này chưa làm.

## TC-12 · Chọn vật bằng SAM 2.1 (bấm điểm / kéo khung thô)

- **Bước trên UI:** mở frame `vid-dash3s-18b2fe_004` → `✨ Chọn vật` (`M`) → bấm vào thân xe tải, hoặc kéo một khung
  thô quanh xe → *Lưu box*.
- **API tương đương:** `POST /api/v1/frames/{id}/segment` với `{"points": [[x, y]]}` hoặc `{"points": [], "box": [...]}`.
- **Mong đợi:** box ôm sát xe tải, không cần vẽ tay; lần bấm sau trên cùng ảnh trả về gần như tức thì.
- **Output thực tế** (CPU, model `sam2.1_hiera_tiny`; lần đầu mỗi ảnh mã hoá khoảng 2 giây, sau đó dùng cache):

```text
điểm (900, 400)             -> bbox [659, 206, 996, 459]  score 0.87   engine sam2-onnx  0.07 s
khung thô [700,180,1030,480] -> bbox [661, 208, 997, 457]  score 0.954  engine sam2-onnx  0.08 s
điểm (800, 300)             -> bbox [684, 287, 839, 343]  score 0.958  engine sam2-onnx  0.07 s   (chỉ dòng chữ trên thùng xe)
```

- **Kết luận:** Đạt. Bấm điểm và kéo khung cho cùng một box (lệch ≤ 2 px). Bấm trúng một chi tiết của vật (dòng 3) thì
  SAM chỉ tách chi tiết đó; khi ấy bấm thêm điểm thứ hai hoặc kéo khung thô.

## TC-13 · Nhiều người dùng: đăng nhập và khoá frame

- **Bước trên UI:** chạy server với `AUTH_REQUIRED=1` → Kien đăng ký (người đầu tiên là admin) và mở một frame → Huy
  đăng nhập trên trình duyệt khác, mở cùng frame.
- **Mong đợi:** chưa đăng nhập thì API từ chối; frame Kien đang mở thì Huy chỉ xem được, hàng đợi hiện 🔒 tên Kien.
- **Output thực tế:**

```text
GET  /frames (chưa đăng nhập)          -> 401 {"code":"LOGIN_REQUIRED","message":"Cần đăng nhập"}
POST /auth/register kien@example.com   -> {"id":"u-278450b6","name":"Kien","admin":true}
POST /auth/register huy@example.com    -> {"id":"u-9c2c0128","name":"Huy","admin":false}
POST /frames/vid-dash3s-18b2fe_004/lock (Kien) -> {"locked":true,"user_id":"u-278450b6"}
POST /frames/vid-dash3s-18b2fe_004/lock (Huy)  -> 409 {"code":"FRAME_LOCKED","message":"Người khác đang mở frame này"}
POST /frames/vid-dash3s-18b2fe_004/actions (Huy, keep) -> 409 {"code":"FRAME_LOCKED", ...}
GET  /presence (Huy) -> "locks":{"vid-dash3s-18b2fe_004":{"user_id":"u-278450b6","name":"Kien"}}
```

- **Kết luận:** Đạt. Mời qua link, vai trò và chia việc được kiểm bằng test tự động (`tests/test_api/test_auth_api.py`).

## TC-14 · So sánh tracker bằng TrackEval

- **Lệnh:** `python tools2d\dam4sam.py ... --stride 3` (nhãn gốc ở keyframe đóng vai nhãn người, lan truyền tối đa 10
  keyframe; lệnh đầy đủ ở cuối `eval/results/tracking.md`).
- **Output thực tế** (3 scene dev, RTX 4050):

```text
flow + ByteTrack (mặc định)  đúng 846  sai 209  đổi ID 32  mất 201   HOTA 0.580  IDF1 0.757    62 s
flow + BoT-SORT              đúng 846  sai 209  đổi ID 28  mất 203   HOTA 0.580  IDF1 0.758    98 s
lai flow + DAM4SAM           đúng 859  sai 255  đổi ID 67  mất 173   HOTA 0.562  IDF1 0.730   506 s
DAM4SAM + ByteTrack          đúng 849  sai 261  đổi ID 20  mất 200   HOTA 0.574  IDF1 0.740   848 s
DAM4SAM + BoT-SORT           đúng 856  sai 248  đổi ID 18  mất 196   HOTA 0.578  IDF1 0.747   731 s
DAM4SAM thuần                đúng 1007 sai 691  đổi ID 83  mất 0     HOTA 0.532  IDF1 0.685  1029 s
```

  Held-out 20 scene (795 keyframe):

```text
flow + ByteTrack (mặc định)  đúng 3438  sai 1161  đổi ID 171  mất 853   HOTA 0.563  IDF1 0.705    731 s
flow + BoT-SORT              đúng 3460  sai 1146  đổi ID 161  mất 853   HOTA 0.564  IDF1 0.709   1090 s
DAM4SAM + BoT-SORT           đúng 3678  sai 1315  đổi ID 155  mất 639   HOTA 0.573  IDF1 0.725   2456 s
```

- **Kết luận:** Đạt (phép đo chạy được và cho kết luận rõ). Mặc định giữ flow + ByteTrack (CPU, nhanh). DAM4SAM + BoT-SORT
  hơn một chút trên held-out (HOTA +0.010, nhãn đúng +7%, mất dấu −25%, box sai +13%, thời gian ×3,4) nên là luồng
  "Chính xác" người dùng tự chọn; trên dev hai luồng ngang nhau.
