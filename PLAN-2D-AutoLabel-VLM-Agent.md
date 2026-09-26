# Plan AutoLabel 2D — Ensemble pre-label + QA Agent đa tín hiệu + Review by exception

Phạm vi: 2D box + mask trên CAM_FRONT của nuScenes v1.0-mini. Tuần 1–4 theo PRD, người phụ trách chính là P1 (ML 2D).
Ràng buộc từ PRD: không train/fine-tune; mọi nhãn phải được người approve; GPU Colab/Kaggle.

---

## 1. Ý tưởng cốt lõi

Rule "confidence thấp thì cho người xem" là cách ai cũng làm, và nó yếu vì confidence của YOLO không đáng tin trên dữ liệu khác COCO.
Plan này thay nó bằng **ba nguồn kiểm chứng độc lập mà project đã có sẵn nhưng thường bị bỏ phí**:

| Nguồn kiểm chứng | Bắt được lỗi gì | Vì sao project này có |
| :--- | :--- | :--- |
| **Model khác** (ensemble YOLO + open-vocab) | FP của từng model, sai lớp | Chạy thêm 1 model, không cần train |
| **LiDAR** (chiếu point cloud xuống ảnh) | Box "ma" (phản chiếu, poster, bóng), sai kích thước so với khoảng cách, sai lớp | nuScenes có LiDAR + calibration đồng bộ |
| **Thời gian** (tracking qua các frame camera 12Hz) | Box nhấp nháy (FP), object bị sót ở keyframe (FN) | nuScenes có camera sweeps 12Hz giữa các keyframe 2Hz |

Agent gộp ba tín hiệu này thành **risk score cho từng object**. Giao diện dùng risk score để làm **review by exception**:
người vẫn duyệt 100% nhãn, nhưng nhãn rủi ro thấp được duyệt theo lô (1 click cho cả nhóm), còn thời gian của người
dồn vào số ít nhãn rủi ro cao. Đây là chỗ thực sự giảm thời gian (M1) mà không vi phạm human-in-the-loop.

Điểm hay thứ hai: cách kiểm chứng bằng LiDAR ở bước 2D chính là nền móng cho ghép cặp 2D↔3D ở phần 3D (FR-07), nên không làm hai lần.

---

## 2. Pipeline

```
Frame (CAM_FRONT keyframe + sweeps 12Hz + LIDAR_TOP + calib)
   │
   ├─[A] Anonymize ── EgoBlur (mặt + biển số) ── ảnh đã làm mờ mới được ra khỏi server
   │
   ├─[B] Pre-label ensemble (chạy trên ảnh gốc, offline batch, cache theo frame)
   │     ├─ YOLO11x           → car, truck, bus, pedestrian, bicycle, motorcycle
   │     ├─ Open-vocab (Grounding DINO / OWLv2 / YOLO-World, chọn 1 sau benchmark tuần 1)
   │     │                     → barrier, traffic_cone, trailer, construction_vehicle (+ toàn bộ lớp để đối chiếu)
   │     └─ Weighted Boxes Fusion → box hợp nhất + "agreement" (bao nhiêu model thấy)
   │
   ├─[C] Mask ── SAM2, prompt bằng box đã fuse
   │
   ├─[D] QA Agent (deterministic, không cần LLM)
   │     ├─ Ensemble check
   │     ├─ LiDAR geometry check
   │     ├─ Temporal check (tracking trên sweeps)
   │     ├─ Mask sanity check
   │     └─ → issue codes + risk score + gợi ý hành động
   │
   ├─[E] (tùy chọn, tuần 4) VLM verifier — chỉ chạy trên crop của object risk cao, hỏi đúng 1 câu về lớp
   │
   └─[F] UI triage → người sửa/duyệt → log chỉnh sửa → calibration + xếp hạng frame khó
```

---

## 3. Chi tiết từng khối

### [A] Ẩn danh
- Dùng **EgoBlur** (Meta, open-source, làm riêng cho mặt và biển số trong dữ liệu góc nhìn xe/người). Không cần tự làm detector.
- Detection chạy trên ảnh gốc; chỉ ảnh đã làm mờ được gửi về client (FR-03).

### [B] Pre-label ensemble
- **Ánh xạ lớp** ghi ra file config (PRD yêu cầu taxonomy không hardcode): `person→pedestrian`, v.v.
- **Open-vocab** lấp 4 lớp mà COCO không có. Prompt văn bản ví dụ: `"traffic cone"`, `"concrete barrier . road barrier"`, `"trailer"`, `"excavator . construction vehicle"`.
- **WBF** (thư viện `ensemble_boxes`) gộp box của các model. Giữ thêm các trường:
  - `agreement`: số model phát hiện ra object (1 hay 2)
  - `score_per_model`: score gốc của từng model
- Ngưỡng score để thấp (~0.1–0.2) để ưu tiên recall: box thừa rẻ để xóa, box thiếu thì phải vẽ tay nên đắt.

### [C] Mask
- SAM2 với prompt box. Mỗi object nhận 1 mask và `sam_score`.

### [D] QA Agent — bảng kiểm tra

| Code | Tín hiệu | Cách tính | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `SINGLE_MODEL` | Ensemble | `agreement == 1` | Chỉ 1 model thấy → dễ là FP |
| `CLASS_CONFLICT` | Ensemble | Hai model đồng ý vị trí (IoU ≥ 0.5) nhưng khác lớp | Cần người chọn lớp |
| `NO_LIDAR_SUPPORT` | LiDAR | Số điểm LiDAR chiếu rơi vào box < k, trong khi box đủ lớn (object gần) | Box "ma": phản chiếu, poster, bóng |
| `SIZE_DEPTH_MISMATCH` | LiDAR | Chiều cao thật ước lượng = `h_px · depth_median / f_y`, so với khoảng hợp lý của lớp (pedestrian 1.0–2.2 m, car 1.2–2.2 m…) | Sai lớp hoặc box sai kích thước |
| `FLICKER` | Thời gian | Track xuất hiện ở keyframe nhưng không có ở ≥ 2 sweep liền kề | FP ngẫu nhiên |
| `RECOVERED_BY_TRACK` | Thời gian | Track có ở sweep trước và sau nhưng detector sót ở keyframe → đề xuất box nội suy | **Bắt FN**, điểm yếu lớn nhất của pre-label |
| `MASK_BAD` | Mask | `mask_area / box_area` ngoài [0.2, 0.95] hoặc mask vỡ thành nhiều mảnh | Mask cần sửa |
| `HEAVY_OVERLAP` | Hình học | IoU với box cùng lớp > 0.7 sau fusion | Trùng lặp |

Ghi chú:
- Không có rule "chạm biên thì xóa": object bị cắt ở biên vẫn là nhãn hợp lệ trong nuScenes.
- Chiếu LiDAR xuống ảnh dùng sẵn `NuScenesExplorer.map_pointcloud_to_image` của devkit. Code này dùng lại được cho FR-07.
- Tracking: ByteTrack chạy trên toàn bộ sweeps 12Hz (YOLO chạy đủ nhanh cho việc này), sau đó chỉ giữ kết quả ở keyframe.

**Risk score** cho mỗi object: bắt đầu bằng tổng có trọng số của các issue cộng `(1 − calibrated_score)`. Trọng số ban đầu đặt bằng nhau
và chỉnh lại ở tuần 4 theo dữ liệu chỉnh sửa thật (giống cách PRD chỉnh α, β, γ cho điểm độ khó).
**Risk score của frame** = max hoặc tổng risk của các object → dùng trực tiếp cho active learning (FR-26).

### [E] VLM verifier (tùy chọn, có điều kiện)
- Chỉ chạy trên **crop** của object có `CLASS_CONFLICT` hoặc `SIZE_DEPTH_MISMATCH`, và object thuộc các lớp open-vocab.
- Chỉ hỏi câu đóng: *"Vật thể trong khung là gì? Chọn 1: traffic_cone / barrier / pedestrian / other."* Không để VLM vẽ box.
- Model: VLM nhỏ chạy local, lượng tử 4-bit trên Colab (ví dụ họ Qwen-VL), hoặc API nếu nhóm có ngân sách.
- **Điều kiện giữ lại:** trên tập eval, VLM phải làm giảm số lần người đổi lớp của nhóm object này. Không giảm thì bỏ, và ghi kết quả âm tính vào báo cáo (đó vẫn là một kết quả có giá trị).

### [F] UI triage và vòng phản hồi
- Object chia 3 nhóm theo risk: **thấp** (thu gọn, có nút "Duyệt tất cả nhóm này"), **trung bình**, **cao** (tự mở, hiện issue code + lý do).
- Box `RECOVERED_BY_TRACK` hiển thị bằng nét đứt, phải được người xác nhận thì mới thành nhãn.
- Mọi thao tác đều được log: giữ / sửa (IoU trước và sau) / xóa / đổi lớp / thêm mới.
- **Calibration:** dùng isotonic regression theo từng lớp trên cặp (score, "nhãn được giữ mà không sửa"), refit sau mỗi 50 object. Kết quả dùng cho ngưỡng hiển thị và cho risk score. Báo cáo ECE trước và sau calibration.
- Không retrain (PRD). Vòng lặp "học" chỉ gồm calibration, trọng số risk và xếp hạng frame.

---

## 4. Ground truth 2D và cách đánh giá

- nuScenes **không có 2D box gốc**. Lấy GT bằng script devkit `export_2d_annotations_as_json.py` (chiếu 3D box xuống ảnh rồi lấy hộp bao).
  - Hộp bao của 8 đỉnh chiếu thường **rộng hơn** box sát vật thể của detector, nên báo cáo mAP@0.5 là chính; số @0.7 thấp là chuyện bình thường, không phải lỗi. Ghi rõ điều này trong báo cáo.
  - Lọc GT theo `visibility` (bỏ mức 0–40%) và theo số điểm LiDAR (> 0), nếu không sẽ phạt detector vì không thấy object gần như bị che hoàn toàn.
- Nếu muốn đánh giá mask thật thì dùng thêm vài trăm ảnh **nuImages** (có box + mask 2D gốc).

**Đánh giá QA Agent trước khi có UI** (quan trọng, làm được từ tuần 3):
coi GT là "người sửa giả lập". Một object pre-label được tính là "cần sửa" nếu không khớp GT hoặc IoU với GT < 0.9 hoặc sai lớp (cùng định nghĩa M4).
Sau đó đo:
- **Flag recall**: trong số object cần sửa, bao nhiêu % bị agent đánh dấu risk cao/trung bình. Chỉ số này cho biết người có bỏ sót lỗi không nếu duyệt nhanh nhóm rủi ro thấp.
- **Flag precision**: trong số object bị đánh dấu, bao nhiêu % thật sự cần sửa. Chỉ số này cho biết agent có "báo động giả" làm phí thời gian người không.
- **FN recovery**: số GT bị detector sót nhưng được `RECOVERED_BY_TRACK` đề xuất đúng.

Mục tiêu hợp lý: flag recall cao (≥ 0.9) là ưu tiên số một, vì nhóm rủi ro thấp sẽ được duyệt theo lô.

| Chỉ số | Đo gì | Nguồn |
| :--- | :--- | :--- |
| mAP@0.5 theo lớp | Chất lượng pre-label (từng model và sau fusion) | So với GT chiếu |
| Flag recall / precision | Agent có đúng chỗ không | GT giả lập, rồi log thật |
| FN recovery | Tracking có bù được sót không | GT |
| M4 — tỷ lệ phải sửa | Log UI | PRD |
| M1 — thời gian/frame | A/B: triage vs danh sách phẳng vs thủ công | PRD |
| ECE | Calibration có tác dụng không | Log UI |

A/B nên có **3 nhánh**: thủ công / pre-label không có triage / pre-label có triage. Chỉ khi có nhánh giữa mới tách được
công của model và công của agent. Thiếu nhánh này thì không chứng minh được agent có ích.

---

## 5. Kế hoạch 4 tuần

| Tuần | P1 (ML 2D) làm | Done khi |
| :--- | :--- | :--- |
| 1 | Export GT 2D (lọc visibility); file ánh xạ lớp; benchmark YOLO11x + 2 ứng viên open-vocab trên ~100 keyframe; chọn open-vocab | Bảng mAP@0.5 theo lớp của từng model; đã chọn model; có baseline thời gian thủ công (PRD) |
| 2 | WBF; SAM2; EgoBlur; xuất JSON theo schema PRD (có `agreement`, `score_per_model`, `issues: []`); cache theo frame; notebook batch có checkpoint | Chạy hết 404 keyframe không cần trông; mAP sau fusion ≥ mAP model tốt nhất |
| 3 | Chiếu LiDAR và các check LiDAR; ByteTrack trên sweeps và các check thời gian; risk score; đánh giá agent bằng GT giả lập | Bảng flag recall/precision theo từng check; biết check nào có ích và check nào nên bỏ |
| 4 | Cùng P4 làm UI triage; log chỉnh sửa; isotonic calibration; (tùy chọn) thử VLM verifier; chạy A/B 3 nhánh | Có số cho M1, M4, flag recall trên log thật; quyết định giữ hay bỏ VLM dựa trên số |

Giao diện với các thành viên khác:
- **P3 (Backend):** chốt JSON schema vào cuối tuần 1; agent chạy như một job trong queue, kết quả cache theo `frame_id` và phiên bản model.
- **P4 (Frontend):** cần 3 thứ, là nhóm risk, issue code kèm câu giải thích, và box nét đứt cho `RECOVERED_BY_TRACK`.
- **P2 (ML 3D):** dùng chung hàm chiếu LiDAR→ảnh; sau này 3D box từ detector 3D thay thế "LiDAR thô" trong các check, cho ra đúng phép ghép cặp FR-07.

---

## 6. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
| :--- | :--- | :--- |
| Open-vocab yếu với cone/barrier nhỏ ở xa | AP của 2 lớp này rất thấp ở tuần 1 | Chạy open-vocab trên ảnh phóng to (tiling); chấp nhận và báo cáo tách lớp |
| Check LiDAR sai ở object xa (ít điểm) | `NO_LIDAR_SUPPORT` có precision thấp | Chỉ áp dụng khi depth ước lượng < 40 m hoặc box đủ lớn |
| Tracking ở 12Hz tốn GPU | Batch chậm | Chỉ track quanh keyframe (±3 sweep) |
| Agent báo động giả nhiều, người bỏ qua cảnh báo | Flag precision < 0.3 | Bỏ check kém nhất; hiển thị tối đa 1 issue quan trọng nhất cho mỗi object |
| Nhóm rủi ro thấp chứa lỗi thật | Flag recall < 0.9 | Hạ ngưỡng nhóm thấp; không bật "duyệt cả nhóm" cho lớp có recall kém |
| GT chiếu từ 3D lệch với box sát vật thể | mAP@0.7 thấp bất thường | Báo cáo @0.5 là chính, giải thích rõ lý do |

---

## 7. Quyết định cần chốt trong tuần 1

1. Model open-vocab nào (theo số benchmark, không chọn theo độ nổi tiếng).
2. k (số điểm LiDAR tối thiểu) và bảng chiều cao hợp lý theo lớp.
3. Ngưỡng chia 3 nhóm risk, và có cho phép "duyệt cả nhóm" ngay từ đầu không hay chỉ bật sau khi đo flag recall.
4. VLM verifier có nằm trong phạm vi không, hay để làm thí nghiệm phụ ở tuần 4.
