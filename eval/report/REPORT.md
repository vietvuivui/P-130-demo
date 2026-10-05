# Report đánh giá mô hình — AutoLabel 3D

Cập nhật 2026-10-04 · nhóm P-130 · biểu đồ: notebook [figures.ipynb](figures.ipynb) (matplotlib, hiện ngay trong notebook)

Report gom mọi mô hình đã thử nghiệm, các chỉ số đánh giá và các phương pháp tối ưu cho hai phần:

- **3D**: gán box 3D trên LiDAR.
- **2D**: gán box trên ảnh camera trước.

Mỗi phần có thêm lan truyền nhãn qua video, QA Agent và thời gian suy luận. Mọi số đo đều so với **nhãn gốc của
nuScenes** và đọc từ các file trong `eval/results/`; nguồn của từng bảng ghi ở cuối report.

## Tóm tắt

| Phần | Cấu hình đang dùng | Kết quả trên tập test | Thời gian |
|---|---|---|---|
| Detector 3D | Gộp 4 mô hình LiDAR + tinh chỉnh theo track | **mAP 0.668 · NDS 0.713** · P 0.777 · R 0.809 · F1 0.793 | 2.56 s / keyframe (GPU) |
| Detector 2D | YOLOE-26-L fine-tune toàn mạng trên nuImages | **mAP50 0.589** · P 0.487 · R 0.806 · F1 0.607 | 0.069 s / ảnh (GPU) |
| Lan truyền 2D | Optical flow + ghép kiểu ByteTrack | P 0.773 · R 0.711 · F1 0.741 · đổi ID 76 | 1.39 s / lần lan truyền |
| Lan truyền 3D | Vận tốc mô hình + giữ track 4 keyframe | P 0.940 · R 0.653 · F1 0.770 | ≈ 0, không chạy mô hình |
| QA Agent 3D | Kiểm chứng box bằng 6 camera | 51% box được tự duyệt, 93% trong số đó đúng; bắt 86% box sai | — |

So với điểm xuất phát:

- **3D:** mAP tăng từ 0.578 lên 0.668 (+0.090) so với CenterPoint voxel, mô hình đơn tốt nhất.
- **2D:** mAP50 tăng từ 0.312 lên 0.589 (+89%) so với YOLOE zero-shot; bản linear probe trước đó đạt 0.366.
- **Lan truyền 2D:** nhãn đúng tăng 12% và đổi ID giảm 60% so với khi không có optical flow.

## Luồng hệ thống hiện tại: mặc định và tuỳ chọn (cập nhật 2026-10-03)

```
Dữ liệu vào (video / bộ ảnh / nuScenes / KITTI)
   │  cắt frame: keyframe 2 Hz + ảnh giữa 10–12 Hz (sweep)
   ▼
① Detector 2D ── YOLOE-26-L fine-tune, 1280 px, giữ box score ≥ 0.30
   │                                   ┌─ ② Detector 3D (dự án có LiDAR): 4 mô hình LiDAR → gộp → tinh chỉnh theo track
   ▼                                   ▼
③ Gộp 2D + 3D ── box 3D chiếu xuống ảnh; box chỉ camera thấy bị hạ điểm ×0.5 (×0.3 nếu không có điểm LiDAR)
   ▼
④ QA Agent ── confidence · LiDAR · temporal (so với sweep t−2…t+2) · hình học → rủi ro low / medium / high
   │           + đề xuất RECOVERED_BY_TRACK (nét đứt) khi người đã xác nhận / vẽ vật ở sweep hai bên
   ▼
⑤ Người duyệt ── review by exception: xem box medium / high, duyệt theo lô box low, thêm box (vẽ hoặc bấm ✨ Chọn vật)
   ▼
⑥ Lan truyền ── nhãn đã duyệt sang các keyframe sau: optical flow dự đoán + ByteTrack ghép với box YOLO
   ▼
⑦ Xuất COCO / nuScenes / KITTI (chỉ frame đã duyệt) · Metrics · TrackEval
```

| Bước | Mặc định (đang chạy) | Tuỳ chọn (có sẵn, đang tắt hoặc người dùng tự chọn) | Vì sao chọn mặc định này |
|---|---|---|---|
| ① Detector 2D | YOLOE-26-L fine-tune nuImages, 1280 px, ngưỡng giữ 0.30 | YOLOE zero-shot (open-vocab, đổi được lớp bằng chữ), YOLO26, YOLO-World, Grounding DINO, Florence-2, gộp nhiều model; chạy 1920 px; lưới ô 2×2 (`yoloe.tiles`); lật ngang (`tta_flip`) | Fine-tune toàn mạng: mAP50 0.312 → 0.589 (linear probe 0.366). 1920 kém hơn (0.344, đo với linear probe). Lưới ô bắt thêm vật nhỏ nhưng box sai +60% (`improve.md`) |
| ② Detector 3D | Gộp CenterPoint voxel + pillar + SSN + PointPillars, tinh chỉnh theo track | Một mô hình đơn; thêm lật trục (TTA, chậm ~4 lần); mô hình camera FCOS3D / PGD | mAP 0.578 (mô hình đơn tốt nhất) → 0.668 |
| ③ Gộp 2D + 3D | Bật; ghép khi IoU ≥ 0.4; box chỉ camera thấy ×0.5 | Tắt gộp; không hạ điểm khi box không có điểm LiDAR (`no_lidar_scale`) | Gộp: việc phải sửa 6990 → 3748. Không hạ điểm: box sai gấp đôi |
| ④ QA temporal | So box keyframe với 4 sweep trực tiếp; đề xuất vật sót (sweep ≥ 0.35) và vật thấy mờ (≥ 0.20, hai bên đều thấy) | Optical flow cho so khớp (`flow`); tính lại score theo sweep (`rescore: mean / linked`); mang nhãn frame trước bằng tracker (`carry_prev`) | Flow / rescore làm lỗi lọt qua duyệt lô tăng. Thấy mờ: bù 108 vật sót thay vì 35 trên held-out, đổi lại 88% đề xuất sai |
| ⑤ Thêm box | Vẽ box, hoặc ✨ Chọn vật bằng SAM 2.1 ONNX (chạy CPU) | SAM 2.1 PyTorch (GPU); GrabCut khi chưa tải model | ONNX không cần GPU: ~2 s lần đầu mỗi ảnh, ~0.1 s các lần bấm sau |
| ⑥ Lan truyền 2D: dự đoán | **Optical flow** ở mọi ảnh (luồng "Nhanh") | **DAM4SAM** (luồng "Chính xác", người dùng chọn trên UI, cần GPU); vận tốc không đổi; lai flow + DAM4SAM | Flow: +14% nhãn đúng so với vận tốc. DAM4SAM hơn flow 0.010 HOTA nhưng chậm 3,4 lần |
| ⑥ Lan truyền 2D: ghép | **ByteTrack** (hai lượt theo score) | BoT-SORT (đi kèm luồng DAM4SAM); OC-SORT (OCR / ORU / OCM); IoU một lượt; không ghép | ByteTrack: box sai −10%. BoT-SORT / OC-SORT không đổi gì đáng kể khi đã có flow |
| ⑥ Lan truyền 3D | Vận tốc của mô hình + ego pose, giữ track 4 keyframe | OC-SORT cho 3D | Giữ 4 keyframe: nhãn đúng +1.6% |
| Nhiều người dùng | Tắt (một người, không đăng nhập) | `AUTH_REQUIRED=1`: tài khoản, link mời, vai trò, chia việc, khoá frame | Giữ cách chạy demo một lệnh |

Hai luồng lan truyền 2D người dùng chọn trên giao diện (held-out 20 scene, chi tiết `eval/results/tracking.md`):

| Luồng | Dự đoán | Ghép | HOTA | IDF1 | Nhãn đúng | Box sai | Mất dấu | Thời gian |
|---|---|---|---|---|---|---|---|---|
| Nhanh (mặc định) | optical flow | ByteTrack | 0.563 | 0.705 | 3438 | 1161 | 853 | ×1, CPU |
| Chính xác | DAM4SAM (SAM 2.1, stride 3) | BoT-SORT | 0.573 | 0.725 | 3678 | 1315 | 639 | ×3,4, GPU |

Trong cả hai luồng, YOLO không tham gia bước dự đoán: nó đã chạy từ bước ① và tracker chỉ đọc box của nó để ghép. Box
ghi ra ở keyframe sau là box YOLO mang ID và lớp của vật đã duyệt.

### Mô tả chi tiết từng bước

**Bài toán và cách tiếp cận.** Xe tự lái cần dữ liệu đã gán nhãn: mỗi ảnh camera và mỗi lần quét LiDAR phải có hộp (box) bao từng
xe, người, cọc tiêu… kèm tên lớp. Gán tay rất tốn công. AutoLabel 3D để máy gán trước, tự chấm xem box nào đáng ngờ,
rồi người chỉ xem lại những box đáng ngờ đó ("review by exception") thay vì xem tất cả.

**Thuật ngữ.**

| Thuật ngữ | Nghĩa |
|---|---|
| Box 2D / box 3D | Hình chữ nhật trên ảnh / hình hộp trong không gian (có vị trí, kích thước, hướng) bao một vật |
| LiDAR | Cảm biến quét laser quanh xe, cho ra đám mây điểm 3D; biết chính xác khoảng cách nhưng thưa dần ở xa |
| Keyframe, sweep | Dữ liệu được ghi ~12 lần / giây. Keyframe là các thời điểm cách nhau 0,5 giây được chọn để gán nhãn; sweep là các ảnh ở giữa, không gán nhãn nhưng dùng làm bằng chứng |
| Score | Độ tự tin của mô hình với một box, từ 0 đến 1 |
| IoU | Mức chồng lấn của hai box (0 = rời nhau, 1 = trùng khít); dùng để quyết định hai box có phải cùng một vật |
| Nhãn gốc (GT) | Nhãn do người của nuScenes gán, dùng làm đáp án để chấm |

**① Detector 2D — tìm vật trên ảnh.** Một mô hình nhận diện (YOLOE-26-L) nhìn từng ảnh và vẽ box quanh vật thuộc 10
lớp của nuScenes. Mô hình gốc được huấn luyện thêm ("fine-tune") trên bộ ảnh đường phố nuImages để quen với cảnh lái
xe. Nó chạy trên cả keyframe lẫn 4 sweep quanh mỗi keyframe. Mọi box score ≥ 0.10 được lưu lại, nhưng chỉ box ≥ 0.30
mới thành nhãn: ngưỡng thấp hơn thì bắt được nhiều vật hơn nhưng người phải xoá nhiều box sai hơn.

**② Detector 3D — tìm vật trong đám mây điểm, rồi "gộp 4 mô hình".** Chỉ chạy khi dữ liệu có LiDAR.

- *Vì sao 4 mô hình.* Mỗi mô hình 3D (CenterPoint voxel, CenterPoint pillar, SSN, PointPillars) chia không gian và học
  theo cách khác nhau nên sai ở những chỗ khác nhau: cái giỏi xe lớn, cái giỏi vật nhỏ. Hỏi cả bốn rồi lấy ý kiến chung
  thì chính xác hơn hỏi một.
- *Gộp thế nào.* Với mỗi keyframe, các box cùng lớp của bốn mô hình có tâm cách nhau dưới một bán kính (xe con 1 m,
  người 0,5 m, cọc tiêu 0,4 m…) được coi là cùng một vật; mỗi mô hình góp tối đa một box. Box gộp lấy vị trí, kích
  thước, hướng, vận tốc là trung bình có trọng số theo score. Score mới = score trung bình × tỉ lệ mô hình đồng ý: vật
  cả 4 mô hình cùng thấy giữ nguyên score, vật chỉ 1 mô hình thấy còn 1/4, nên box "một mình một ý" tự tụt xuống dưới.
- *Tinh chỉnh theo track.* Một vật thật không đổi kích thước hay quay ngoắt 180° giữa hai keyframe. Nên các box của cùng
  một vật qua các keyframe được nối thành chuỗi (track), rồi: kích thước lấy trung vị cả chuỗi; hướng bị ngược đầu đuôi
  thì lật lại theo đa số; vận tốc tính lại từ quãng đường thật; keyframe bị hụt box ở giữa chuỗi thì nội suy thêm.
- *Kết quả.* mAP (độ chính xác trung bình, thang 0–1) từ 0.578 của mô hình đơn tốt nhất lên 0.668, không phải huấn luyện
  thêm gì. Giá phải trả là thời gian: chạy bốn mô hình thay vì một.

**③ Gộp 2D + 3D — dùng LiDAR để sửa nhãn trên ảnh.** Box 3D được chiếu xuống ảnh camera thành box 2D. Box của camera
trùng với box chiếu (IoU ≥ 0.4, cùng lớp) thì lấy box chiếu và score cao hơn trong hai. Box chỉ camera thấy, LiDAR không
xác nhận, bị nhân score với 0.5, nên phần lớn rơi xuống dưới ngưỡng giữ. Lý do: LiDAR đo được khoảng cách thật nên ít
"tưởng tượng" ra vật hơn camera. Trên tập test, số box người phải sửa (thừa + sót) giảm từ 6990 xuống 3748.

Đo lại ngày 04/10 với detector fine-tune toàn mạng, trên **cả 6 camera** của tập test (5742 ảnh, 25991 vật). Từ ngày
này box chỉ camera thấy mà không có điểm LiDAR nào trong box bị nhân 0.3 thay vì 0.5 (95% box loại này không khớp nhãn
gốc):

| Cấu hình | mAP50 | P | R | F1 | Box thừa | Vật sót | Phải sửa |
|---|---|---|---|---|---|---|---|
| Chỉ detector ảnh | 0.560 | 0.534 | 0.763 | 0.628 | 17262 | 6170 | 23432 |
| Gộp box 3D, mọi box chỉ camera x0.5 (trước 04/10) | 0.671 | 0.628 | 0.818 | 0.710 | 12612 | 4736 | 17348 |
| **Gộp box 3D, box 0 điểm LiDAR x0.3 (mặc định mới)** | **0.672** | **0.672** | 0.812 | **0.735** | 10287 | 4897 | **15184 (−35%)** |

Gộp có lợi ở cả 6 camera (mAP50 +0.08 đến +0.14). Riêng CAM_FRONT: mAP50 0.589 → 0.700, phải sửa 7023 → 3504. Bảng từng
camera: `eval/results/lidar2d.md`.

**④ QA Agent — máy tự chấm box nào đáng ngờ.** Mỗi box đi qua bốn nhóm kiểm tra; mỗi kiểm tra không đạt sinh một mã
lỗi kèm lời giải thích cho người duyệt:

| Nhóm | Hỏi gì | Mã lỗi |
|---|---|---|
| Độ tự tin | Score có thấp không? Mô hình có phân vân giữa hai lớp không? | `LOW_CONFIDENCE`, `CLASS_CONFLICT` |
| LiDAR | Trong box có điểm LiDAR không? Kích thước box có hợp với khoảng cách đo được không? | `NO_LIDAR_SUPPORT`, `SIZE_DEPTH_MISMATCH` |
| Thời gian | Vật có xuất hiện lại ở các sweep ngay trước và sau không, hay chỉ loé lên một ảnh? | `FLICKER`, `RECOVERED_BY_TRACK` |
| Hình học | Box có to bất thường, hay tỉ lệ rộng / cao lạ so với lớp không? | `BOX_TOO_LARGE`, `ASPECT_RATIO_ABNORMAL` |

Bốn nhóm được cộng có trọng số thành một điểm rủi ro 0–1, chia ba mức: **low** (xanh, < 0.17), **medium** (vàng),
**high** (đỏ, ≥ 0.40; ngưỡng dò lại ngày 04/10, xem mục 4). Box có bất kỳ mã lỗi nào thì không bao giờ được xếp low. `RECOVERED_BY_TRACK` là trường hợp
ngược: detector bỏ sót (hoặc thấy quá mờ) ở keyframe nhưng các sweep hai bên đều thấy, nên máy đề xuất một box nét đứt
để người xác nhận.

**⑤ Người duyệt.** Hàng đợi xếp frame khó nhất lên đầu. Trong mỗi frame, người duyệt xem box đỏ và vàng (giữ, xoá, đổi
lớp, sửa box), rồi bấm một nút để duyệt cả nhóm box xanh. Vật máy sót thì vẽ box, hoặc bấm ✨ Chọn vật rồi bấm vào vật:
mô hình SAM 2.1 tự tách vật ra và tạo box. Frame chỉ được duyệt khi không còn box nào chưa xử lý.

**⑥ Lan truyền nhãn — không phải duyệt lại cùng một vật ở mọi frame.** Một chiếc xe có mặt liên tục vài giây, tức hàng
chục keyframe. Sau khi người duyệt một keyframe, tracker mang từng nhãn sang các keyframe sau, qua hai bước lặp lại ở
mỗi ảnh:

- *Dự đoán:* vật này ở ảnh kế tiếp nằm đâu? Mặc định dùng **optical flow**: tính mỗi điểm ảnh dịch đi bao nhiêu giữa
  hai ảnh liên tiếp rồi dời box theo. Tuỳ chọn **DAM4SAM**: mô hình SAM 2.1 nhớ hình dạng vật và tô lại vật ở ảnh mới.
- *Ghép:* box dự đoán ứng với box nào detector vừa thấy ở ảnh đó? Mặc định dùng **ByteTrack**: ghép với box score cao
  trước, box score thấp chỉ dùng cho vật chưa ghép được, để box mờ không "cướp" mất vật rõ.

Nhãn lan truyền mang tag ↦ và một độ tin cậy riêng; vật không ghép được trong 6 ảnh liên tiếp thì ngừng theo dõi. Vật
người đã xoá cũng được nhớ để tự xoá ở frame sau. Với 3D, vật được dời theo vận tốc mô hình đã đo và chuyển động của
chính xe thu dữ liệu.

**⑦ Xuất và đo.** Chỉ frame người đã duyệt mới được xuất (COCO, nuScenes, KITTI), kèm nhật ký mọi lần sửa. Từ nhật ký,
hệ thống tính tỉ lệ nhãn máy phải sửa và thời gian duyệt mỗi frame, tức các số nói lên máy đã đỡ được bao nhiêu công.

## 1. Giao thức đánh giá và định nghĩa chỉ số

**Tách dữ liệu để tránh overfit.** Cách làm được chọn trên **dev**, gồm 3 scene demo của UI (scene-0035, 0097, 0101;
119 keyframe). Số liệu báo cáo lấy trên **test**, gồm các scene val khác của nuScenes trainval chưa dùng để chọn gì:

| Tập test | Số scene | Số keyframe |
|---|---|---|
| Detector 3D | 24 | 957 |
| Detector 2D | 24 | 957 (CAM_FRONT) |
| Lan truyền 2D, P/R/F1 của 3D | 20 | 795 |

Tham số lấy theo mặc định của bài báo gốc, không dò trên nhãn gốc.

| Chỉ số | Ý nghĩa |
|---|---|
| **mAP** (3D) | AP trung bình 10 lớp, chuẩn nuScenes: ghép box theo khoảng cách tâm trên mặt phẳng BEV, lấy trung bình ở 4 ngưỡng 0.5 / 1 / 2 / 4 m |
| **NDS** (3D) | nuScenes Detection Score = ½·mAP + ½·trung bình của (1 − sai số) qua 5 sai số TP: vị trí, kích thước, hướng, vận tốc, thuộc tính |
| **mATE / mASE / mAOE / mAVE** | Sai số trung bình của box đúng: vị trí (m), kích thước (1 − IoU sau khi căn tâm và hướng), hướng (rad), vận tốc (m/s). Thấp hơn là tốt |
| **mAP50 / mAP70** (2D) | AP trung bình các lớp, box tính là đúng khi IoU ≥ 0.5 (hoặc 0.7). Nhãn 2D là hình chiếu của box 3D gốc |
| **Precision (P)** | Box đúng / box ghi ra |
| **Recall (R)** | Box đúng / số vật trong nhãn gốc |
| **F1** | 2·P·R / (P + R) |
| P/R/F1 của 3D | Cùng lớp, tâm BEV ≤ 2 m, score ≥ 0.3; chỉ tính vật có ít nhất 1 điểm LiDAR/radar và nằm trong tầm theo lớp như nuScenes (xe 50 m, người / xe 2 bánh 40 m, cone / barrier 30 m) |
| P/R/F1 của 2D | IoU ≥ 0.5, ngưỡng giữ box của sản phẩm (0.30) |
| **Lan truyền nhãn** | Nhãn gốc ở keyframe 0, 5, 10… đóng vai nhãn người đã duyệt, lan truyền tiếp tối đa 10 keyframe |
| Đúng / sai / đổi ID / thiếu | Đúng: box ghi ra nằm trên đúng vật. Sai: box không trùng vật nào. Đổi ID: box nhảy sang vật khác. Thiếu: vật còn trong ảnh nhưng không có box |
| **Điểm lan truyền** | Đúng − sai − 2 × đổi ID. Đổi ID bị phạt gấp đôi vì box mang lớp và kích thước của vật cũ, người duyệt khó phát hiện |
| **Thời gian** | GPU = laptop RTX 4050; CPU = 2 luồng x86. Thời gian 3D tính cả nạp dữ liệu |

## 2. Detector 3D

Tất cả là trọng số MMDetection3D có sẵn, không train lại và không dùng TTA.

Biểu đồ: [mAP và NDS của mô hình 3D](figures.ipynb) (mục 1 trong notebook).

| Mô hình | Cảm biến | mAP | NDS | mATE (m) | mASE | mAOE (rad) | mAVE (m/s) | P | R | F1 | F1 tốt nhất (ngưỡng) | s / keyframe | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FCOS3D | camera | 0.336 | 0.421 | 0.682 | 0.231 | 0.451 | 1.725 | 0.823 | 0.440 | 0.573 | 0.573 (0.3) | 2.72 | 0.55 GB |
| PGD | camera | 0.399 | 0.458 | 0.599 | 0.247 | 0.375 | 1.288 | 0.995 | 0.012 | 0.024 | 0.660 (0.1) | 2.71 | 0.56 GB |
| SSN | LiDAR | 0.441 | 0.568 | 0.325 | 0.260 | 0.346 | 0.332 | 0.787 | 0.497 | 0.609 | 0.609 (0.3) | 0.59 | 0.57 GB |
| PointPillars | LiDAR | 0.473 | 0.559 | 0.389 | 0.265 | 0.559 | 0.299 | 0.752 | 0.623 | 0.681 | 0.681 (0.3) | 0.85 | 1.74 GB |
| CenterPoint pillar | LiDAR | 0.528 | 0.610 | 0.292 | 0.258 | 0.369 | 0.360 | 0.560 | 0.845 | 0.673 | 0.749 (0.5) | 0.41 | 0.48 GB |
| CenterPoint voxel | LiDAR | 0.578 | 0.655 | 0.257 | 0.237 | 0.286 | 0.368 | 0.625 | 0.866 | 0.726 | 0.777 (0.5) | 0.62 | 0.50 GB |
| CenterPoint voxel + track | LiDAR | 0.616 | 0.689 | 0.256 | 0.219 | 0.212 | 0.325 | 0.620 | 0.894 | 0.733 | 0.789 (0.5) | 0.63 | 0.50 GB |
| Gộp 4 LiDAR | LiDAR | 0.637 | 0.690 | 0.232 | 0.242 | 0.274 | 0.288 | 0.783 | 0.788 | 0.785 | 0.785 (0.3) | 2.55 | — |
| **Gộp 4 LiDAR + track (mặc định)** | LiDAR | **0.668** | **0.713** | 0.229 | 0.237 | 0.220 | **0.268** | 0.777 | 0.809 | **0.793** | 0.793 (0.3) | 2.56 | — |
| Gộp 4 LiDAR + 2 camera + track | cả hai | 0.662 | 0.711 | 0.236 | 0.234 | **0.207** | 0.277 | 0.876 | 0.671 | 0.760 | 0.760 (0.3) | 8.01 | — |

Cách đọc bảng:

- **mAP, NDS và sai số** đo trên 24 scene test.
- **P / R / F1** đo ở score ≥ 0.3 trên 20 scene test (17251 vật, 795 keyframe). Cột "F1 tốt nhất" là F1 cao nhất trong 4
  ngưỡng 0.05 / 0.1 / 0.3 / 0.5. Vì ngưỡng đó được chọn ngay trên tập test, cột này chỉ để tham khảo.
- **Thời gian** đo trên RTX 4050 với 1076 keyframe. Ensemble là tổng thời gian chạy 4 mô hình cộng bước gộp và track
  (khoảng 0.085 s trên CPU).

Nhận xét:

- **CenterPoint voxel là mô hình đơn tốt nhất** ở mọi chỉ số, trừ tốc độ.
- **Mô hình camera kém xa LiDAR:** mAP 0.34–0.40, sai số vận tốc 1.3–1.7 m/s so với khoảng 0.3 m/s.
- **PGD cho điểm rất thấp:** ở score ≥ 0.3 gần như không ra box (recall 0.012). Mô hình này chỉ dùng được với ngưỡng
  khoảng 0.1.
- **Gộp 4 mô hình LiDAR** làm precision tăng mạnh (0.625 → 0.783) vì box phải được nhiều mô hình đồng ý. Tinh chỉnh theo
  track bù lại recall (0.788 → 0.809) và giảm sai số hướng (0.274 → 0.220 rad).
- **Thêm 2 mô hình camera** tăng precision nhưng giảm mạnh recall, mAP giảm từ 0.668 xuống 0.662, và chậm gấp 3 lần.

Biểu đồ: [Đánh đổi tốc độ – độ chính xác](figures.ipynb) (mục 2 trong notebook).

Biểu đồ: [Precision / Recall / F1 3D](figures.ipynb) (mục 3 trong notebook).

**AP từng lớp** (test 24 scene):

| Mô hình | car | truck | bus | trailer | constr. vehicle | pedestrian | motorcycle | bicycle | traffic cone | barrier |
|---|---|---|---|---|---|---|---|---|---|---|
| FCOS3D | 0.486 | 0.244 | 0.444 | 0.119 | 0.165 | 0.439 | 0.268 | 0.256 | 0.631 | 0.310 |
| PGD | 0.535 | 0.298 | 0.439 | 0.376 | 0.215 | 0.460 | 0.332 | 0.316 | 0.675 | 0.342 |
| SSN | 0.821 | 0.434 | 0.680 | 0.386 | 0.249 | 0.711 | 0.085 | 0.248 | 0.408 | 0.386 |
| PointPillars | 0.826 | 0.357 | 0.538 | 0.629 | 0.202 | 0.771 | 0.124 | 0.308 | 0.573 | 0.401 |
| CenterPoint pillar | 0.851 | 0.516 | 0.682 | 0.508 | 0.278 | 0.840 | 0.269 | 0.240 | 0.686 | 0.414 |
| CenterPoint voxel | 0.858 | 0.501 | 0.707 | 0.319 | 0.318 | 0.882 | 0.496 | 0.435 | 0.803 | 0.458 |
| CenterPoint voxel + track | 0.869 | 0.548 | 0.727 | 0.361 | 0.398 | 0.895 | 0.579 | 0.510 | 0.813 | 0.464 |
| Gộp 4 LiDAR | 0.878 | 0.547 | 0.780 | 0.554 | 0.439 | 0.891 | 0.479 | 0.502 | 0.803 | 0.498 |
| **Gộp 4 LiDAR + track (mặc định)** | **0.886** | **0.587** | 0.803 | **0.620** | **0.473** | 0.900 | 0.555 | 0.561 | 0.802 | 0.495 |
| Gộp 4 LiDAR + 2 camera + track | 0.886 | 0.572 | **0.812** | 0.470 | 0.468 | **0.903** | **0.593** | **0.582** | **0.834** | **0.503** |

Biểu đồ: [AP từng lớp 3D](figures.ipynb) (mục 4 trong notebook).

**Precision / recall từng lớp ở score ≥ 0.3** (test 20 scene):

| | car | truck | bus | trailer | constr. vehicle | pedestrian | motorcycle | bicycle | traffic cone | barrier |
|---|---|---|---|---|---|---|---|---|---|---|
| P · CenterPoint voxel | 0.82 | 0.55 | 0.77 | 0.13 | 0.34 | 0.67 | 0.12 | 0.29 | 0.65 | 0.45 |
| R · CenterPoint voxel | 0.89 | 0.72 | 0.86 | 1.00 | 0.60 | 0.92 | 0.66 | 0.58 | 0.89 | 0.84 |
| P · mặc định | 0.85 | 0.70 | 0.91 | 0.36 | 0.80 | 0.84 | 0.61 | 0.94 | 0.83 | 0.53 |
| R · mặc định | 0.90 | 0.75 | 0.94 | 1.00 | 0.49 | 0.85 | 0.41 | 0.41 | 0.70 | 0.78 |

Ensemble mạnh nhất ở xe lớn: trailer +0.30 AP, xe công trình +0.16, xe đạp +0.13. Với các lớp nhỏ như xe máy, xe đạp và
cone, ensemble đổi recall lấy precision ở cùng ngưỡng score.

## 3. Detector 2D

Biểu đồ: [Chọn detector 2D và fine-tune](figures.ipynb) (mục 5 trong notebook).

| Detector | Tập | mAP50 | mAP70 | P | R | F1 | Thời gian / ảnh (1280 px) |
|---|---|---|---|---|---|---|---|
| YOLO-World v2-L (từ vựng COCO, không có lớp barrier) | 40 keyframe | 0.311 | 0.208 | — | — | — | 3.75 s CPU |
| YOLO26-L (tập lớp COCO, không có lớp barrier) | 40 keyframe | 0.398 | 0.265 | — | — | — | 2.14 s CPU |
| YOLOE-26-L zero-shot | 40 keyframe | **0.452** | **0.278** | 0.506 | 0.616 | 0.556 | 3.31 s CPU |
| YOLOE-26-L zero-shot | dev 119 keyframe | 0.392 | — | 0.566 | 0.578 | 0.572 | |
| YOLOE-26-L linear probe nuImages | dev 119 keyframe | 0.429 | — | 0.525 | 0.633 | 0.574 | |
| YOLOE-26-L fine-tune toàn mạng nuImages | dev 119 keyframe | **0.616** | — | 0.472 | **0.825** | **0.600** | |
| YOLOE-26-L zero-shot | test 957 keyframe | 0.312 | — | **0.499** | 0.523 | 0.511 | như dòng dưới |
| YOLOE-26-L linear probe nuImages | test 957 keyframe | 0.366 | — | 0.484 | 0.581 | 0.528 | 0.076 s GPU |
| **YOLOE-26-L fine-tune toàn mạng nuImages (mặc định)** | **test 957 keyframe** | **0.589** | — | 0.487 | **0.806** | **0.607** | **0.069 s GPU** |

Ghi chú:

- **Bộ 40 keyframe** (scene-0031, 0065) chỉ có nhãn của 6 lớp, nên mAP ở đây cao hơn trên test 10 lớp.
- **Bản fine-tune toàn mạng** (mặc định từ 04/10): 30 epoch ở 1280 px trên nuImages, RTX 3090, mở cả backbone, neck và
  nhánh box; checkpoint chọn theo nuImages val (mAP50 0.782). Trên UI là lựa chọn "Fine-tune full"; "Fine-tune mini" là
  linear probe, "Original" là bản zero-shot.
- **Bản linear probe:** 10 epoch ở 1280 px trên nuImages, chạy trên RTX 3090. Checkpoint chọn theo nuImages
  val (mAP50 0.509). Linear probe chỉ học lại đầu phân lớp, nên kiến trúc và tốc độ không đổi.
- **Thời gian zero-shot trên GPU:** lần chấm đó đọc từ cache detection nên không có số riêng.

Biểu đồ: [AP50 từng lớp 2D](figures.ipynb) (mục 6 trong notebook).

| Lớp | barrier | bicycle | bus | car | constr. | motorcycle | pedestrian | traffic cone | trailer | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| AP50 zero-shot | 0.127 | 0.123 | 0.805 | 0.707 | 0.019 | 0.210 | 0.226 | 0.390 | 0.000 | 0.518 |
| AP50 linear probe | 0.333 | 0.233 | 0.779 | 0.728 | 0.029 | 0.275 | 0.294 | 0.500 | 0.015 | 0.470 |
| AP50 fine-tune toàn mạng | **0.676** | **0.630** | **0.887** | **0.853** | **0.111** | **0.541** | **0.563** | **0.798** | **0.224** | **0.611** |
| Số vật trong nhãn gốc | 744 | 134 | 210 | 2375 | 112 | 81 | 1604 | 908 | 22 | 543 |

Linear probe tăng ở 8/10 lớp (barrier ×2.6, xe đạp ×1.9, cone +0.11), bus và truck giảm nhẹ. Fine-tune toàn mạng tăng ở
cả 10 lớp so với cả hai bản trước; recall 0.523 → 0.806, còn precision gần như không đổi (0.499 → 0.487), tức người
duyệt vẫn phải xoá khoảng một nửa số box ở ngưỡng 0.30. Hai lớp còn yếu là construction vehicle (0.111) và trailer
(0.224, chỉ 22 vật trên test). Hai bản fine-tune chỉ nhận tập lớp đóng gồm 10 lớp nuScenes; dự án dùng lớp khác phải
quay về bản zero-shot (open-vocab).

**Các biến thể TTA đã thử** (dev 119 keyframe, YOLOE zero-shot):

| Biến thể | mAP50 | P | R | F1 |
|---|---|---|---|---|
| **Gốc (prompt config, 1280 px)** | **0.395** | **0.568** | 0.580 | **0.574** |
| Prompt mở rộng theo định nghĩa lớp nuScenes | 0.392 | 0.549 | 0.579 | 0.564 |
| + prompt gây nhiễu | 0.392 | 0.549 | 0.579 | 0.564 |
| + lật ảnh ngang | 0.381 | 0.518 | **0.592** | 0.552 |
| Ảnh 1600 px | 0.365 | 0.520 | 0.555 | 0.537 |

Không biến thể nào tốt hơn cấu hình gốc, nên không dùng.

**Toàn bộ pipeline 2D của sản phẩm.** Gồm detector, fusion, ngưỡng 0.30 và QA, chạy với model zero-shot trên test 20
scene (795 keyframe):

| Chỉ số | Giá trị |
|---|---|
| mAP50 | 0.296 |
| mAP70 | 0.077 |
| Precision | 0.486 |
| Recall | 0.524 |
| F1 | 0.504 |

Chạy lại ngày 04/10 với model fine-tune toàn mạng trên cùng 795 keyframe (`eval/results/temporal/full_heldout20.json`):

| Chỉ số | Zero-shot | Fine-tune toàn mạng |
|---|---|---|
| mAP50 | 0.296 | **0.611** |
| mAP70 | 0.077 | **0.269** |
| Precision | 0.486 | 0.475 |
| Recall | 0.524 | **0.815** |
| F1 | 0.504 | **0.600** |
| Vật sót phải vẽ thêm | 2565 | **994** |
| Box sai phải xoá | 2980 | 4851 |
| Box sai lọt vào nhóm low (duyệt theo lô) | 1632 | 3621 |

Vật sót giảm 61%, đổi lại box sai tăng vì mô hình ra nhiều box hơn mà precision không đổi. QA Agent bắt lỗi kém hơn
với detector mới (tỉ lệ box sai được gắn cờ 45% → 25%), nên ngưỡng risk cần dò lại (xem mục 9). Lan truyền nhãn với
cấu hình mặc định trên cùng tập: nhãn đúng 4074 / 5139 box ghi ra (79.3%), đổi ID 225, box sai 840; chi tiết ở
`eval/results/temporal/report.md`, mục 0b.

## 4. QA Agent

QA Agent chấm rủi ro cho từng box. Box rủi ro thấp được duyệt theo lô, còn người duyệt chỉ xem kỹ các box bị gắn cờ.

Biểu đồ: [QA Agent](figures.ipynb) (mục 10 trong notebook).

**QA 3D: kiểm chứng box bằng camera** (dev 3 scene, mỗi mô hình dùng ngưỡng điểm riêng):

| Mô hình | Số box | Tỉ lệ tự duyệt | Precision nhóm tự duyệt | Recall bắt box sai | Precision cờ | Tỉ lệ box sai | Vật bị sót |
|---|---|---|---|---|---|---|---|
| PointPillars | 2668 | 0.536 | **0.948** | 0.881 | 0.443 | 0.234 | 1514 / 3559 |
| SSN | 3468 | 0.432 | 0.920 | **0.909** | **0.607** | 0.379 | 1405 / 3559 |
| CenterPoint pillar | 3640 | 0.481 | 0.920 | 0.866 | 0.478 | 0.286 | 962 / 3559 |
| **CenterPoint voxel** | 3697 | 0.512 | 0.931 | 0.863 | 0.457 | 0.259 | **818 / 3559** |
| FCOS3D | 4111 | 0.529 | 0.798 | 0.722 | 0.590 | 0.385 | 1029 / 3559 |
| PGD | 3304 | 0.587 | 0.872 | 0.744 | 0.526 | 0.293 | 1222 / 3559 |

Với CenterPoint voxel:

- **Tự duyệt:** một nửa số box được tự duyệt, và 93% trong số đó đúng.
- **Bắt lỗi:** QA bắt được 86% box sai.
- **Vật bị sót:** mô hình này sót ít vật nhất.

**QA 2D: các cấu hình xử lý sau detector** (test 20 scene, cùng detection zero-shot):

| Cấu hình | mAP50 | P | R | F1 | Box xem tay | Box sai lọt duyệt lô | Recall cờ | Precision cờ | s / keyframe |
|---|---|---|---|---|---|---|---|---|---|
| **Mặc định** | 0.296 | 0.486 | 0.524 | 0.504 | 1707 | 1632 | **0.452** | 0.790 | **0.044** |
| + optical flow cho QA temporal | 0.296 | 0.486 | 0.524 | 0.504 | 1059 | 2082 | 0.301 | 0.848 | 0.143 |
| + tính lại score theo sweep (mean) | 0.283 | **0.582** | 0.482 | **0.527** | 648 | 1400 | 0.248 | 0.711 | 0.047 |
| flow + mean | 0.292 | 0.542 | 0.506 | 0.523 | 492 | 1906 | 0.171 | 0.801 | 0.139 |
| flow + linked | **0.304** | 0.453 | **0.546** | 0.495 | 1415 | 2319 | 0.346 | **0.866** | 0.144 |

Cấu hình mặc định giữ recall cờ cao nhất, tức là ít lỗi lọt qua nhóm duyệt lô nhất. "Tính lại score" giảm số box phải
xem tay từ 1707 xuống 648 nhưng làm sót thêm vật, nên chỉ để làm tuỳ chọn.

**QA 2D với detector fine-tune toàn mạng, trước và sau khi dò lại** (04/10, test 20 scene, 795 keyframe, 9241 box):

| Chỉ số | QA cũ | QA mới |
|---|---|---|
| Nhóm low (duyệt theo lô): số box | 7573 (82%) | 3138 (34%) |
| Nhóm low: tỉ lệ sai | 47.8% | **8.9%** |
| Box sai lọt qua duyệt theo lô | 3621 | **280** |
| Box phải xem tay | 1668 | 6103 |
| Recall cờ / precision cờ | 0.254 / 0.737 | **0.942** / 0.749 |
| Nhóm high: số box / tỉ lệ sai | 13 / 92% | 2300 / 95% |
| Đề xuất RECOVERED_BY_TRACK / trùng vật sót | 627 / 37 | 0 / 0 |

Risk cũ xếp box sai không hơn gì score của detector (AUC 0.841 cả hai); risk mới đạt 0.916 nhờ dùng số điểm LiDAR
trong box (box 0 điểm sai 96%) và ngưỡng khớp với score của detector mới. QA không làm đổi mAP / P / R / F1. Đổi lại
người duyệt phải xem 66% số box, vì detector ra 52% box sai so với nhãn gốc. Chi tiết: `eval/results/qa_tuning.md`.

## 5. Lan truyền nhãn 2D: optical flow × ByteTrack × OC-SORT

Bản đầy đủ hơn, gồm BoT-SORT, DAM4SAM, TrackEval (HOTA / IDF1) và vai trò từng thuật toán trong mỗi tổ hợp:
[eval/results/tracking.md](../results/tracking.md). Mục này giữ số liệu đợt đo đầu (detector zero-shot).

Test 20 scene, 154 lần lan truyền, dùng chung detection zero-shot. Chỉ đếm những box sản phẩm thật sự ghi ra.

- P = đúng / box ghi ra.
- R = đúng / (đúng + thiếu).

Biểu đồ: [Lan truyền 2D](figures.ipynb) (mục 7 trong notebook).

| Optical flow | Ghép detection | Đúng | Sai | Đổi ID | Thiếu | P | R | F1 | Điểm |
|---|---|---|---|---|---|---|---|---|---|
| không | không tracker (IoU 1 tầng) | 2866 | 761 | 205 | 1541 | 0.748 | 0.650 | 0.696 | 1695 |
| không | ByteTrack | 2822 | 652 | 194 | 1673 | 0.769 | 0.628 | 0.691 | 1782 |
| có | không tracker | 3220 | 960 | 82 | 1182 | 0.755 | 0.731 | 0.743 | 2096 |
| có | **ByteTrack (mặc định)** | 3192 | **860** | **76** | 1298 | **0.773** | 0.711 | 0.741 | **2180** |
| có | OC-SORT (OCR) | **3254** | 981 | 101 | **1122** | 0.750 | **0.744** | **0.747** | 2071 |
| có | ByteTrack + OC-SORT | 3215 | 868 | 87 | 1265 | 0.771 | 0.718 | 0.743 | 2173 |

Nhận xét:

- **Optical flow là bước quyết định:** F1 tăng từ 0.696 lên 0.743, đổi ID giảm từ 205 xuống 82.
- **Khi đã có flow, F1 của bốn cách ghép gần như bằng nhau** (0.741–0.747).
- **ByteTrack** cho precision cao nhất và ít đổi ID nhất, nên được chọn theo điểm có phạt đổi ID.
- **OC-SORT** phủ thêm vật (recall cao nhất) nhưng đổi ID tăng từ 82 lên 101. Trên dev, các thành phần ORU và OCM không đổi kết
  quả đo được.
- **Thời gian:** ByteTrack và OC-SORT khác nhau dưới 2%. Optical flow chiếm phần lớn thời gian: 0.33 → 1.39 s mỗi lần lan
  truyền trên GPU.

## 6. Lan truyền nhãn 3D

Tracker 3D ghép box theo khoảng cách tâm (kiểu CenterPoint) và dự đoán vị trí bằng vận tốc mô hình cộng ego pose. Bước
này chỉ dùng dự đoán có sẵn, không chạy lại mô hình.

- P = tỉ lệ box đúng vật.
- R = tỉ lệ vật còn trong tầm có nhãn lan truyền đúng.

Biểu đồ: [Lan truyền 3D](figures.ipynb) (mục 8 trong notebook).

| Cấu hình (test 24 scene) | Box ghi ra | Đúng | Sai | Đổi ID | P | R | F1 | Sai số tâm | Điểm dev |
|---|---|---|---|---|---|---|---|---|---|
| Giữ track 2 keyframe (cũ) | 24771 | 23355 | 946 | **470** | **0.943** | 0.643 | 0.764 | **0.189 m** | 3731 |
| Giữ 3 keyframe | 25047 | 23569 | 962 | 516 | 0.941 | 0.648 | 0.768 | 0.190 m | 3741 |
| **Giữ 4 keyframe (mặc định)** | 25242 | 23731 | 968 | 543 | 0.940 | 0.653 | 0.770 | 0.190 m | **3762** |
| Giữ 6 keyframe | 25418 | 23853 | 972 | 593 | 0.938 | 0.656 | 0.772 | 0.191 m | 3765 |
| + OC-SORT OCR | 25428 | 23872 | 980 | 576 | 0.939 | 0.656 | 0.773 | 0.191 m | 3734 |
| + OCR + ORU | 25446 | **23898** | 979 | 569 | 0.939 | **0.657** | **0.773** | 0.191 m | 3733 |
| chỉ ORU | 24802 | 23382 | 949 | 471 | 0.943 | 0.643 | 0.765 | 0.190 m | 3731 |
| chỉ OCM | 24762 | 23321 | 948 | 493 | 0.942 | 0.642 | 0.763 | 0.189 m | 3731 |

Lan truyền theo vận tốc mô hình (held-out 4 scene):

| Cấu hình | Nhãn đúng | Đổi ID |
|---|---|---|
| Chỉ bù chuyển động xe ego | 3031 | 98 |
| + vận tốc mô hình | 3475 | 67 |
| Mặc định | 3339 | 36 |

Giữ 4 keyframe được chọn theo điểm dev. OCR cho F1 cao hơn một chút trên test, nhưng trên dev lại thấp hơn, nên để tắt;
có thể bật ở tab ⚙ Cài đặt.

## 7. Lịch sử tối ưu

Biểu đồ: [Lịch sử tối ưu](figures.ipynb) (mục 9 trong notebook).

| # | Phương pháp | Phần | Chỉ số quyết định | Trước | Sau | Thời gian thêm | Dùng? |
|---|---|---|---|---|---|---|---|
| 1 | YOLOE-26-L thay YOLO-World | 2D | mAP50 (40 keyframe) | 0.311 | 0.452 | −12% (CPU) | ✅ |
| 2 | Fine-tune toàn mạng YOLOE-26-L trên nuImages (linear probe: 0.366) | 2D | mAP50 test | 0.312 | 0.589 | 0 | ✅ |
| 3 | Ngưỡng giữ box 0.30 | 2D | precision (40 keyframe) | 0.34 | 0.51 | 0 | ✅ |
| 4 | TTA: prompt, lật ảnh, 1600 px | 2D | mAP50 dev | 0.395 | 0.365–0.392 | ×1.5–2 (ước tính) | ❌ |
| 5 | Optical flow khi lan truyền | 2D | nhãn đúng / đổi ID | 2866 / 205 | 3220 / 82 | ×4.2 | ✅ |
| 6 | ByteTrack | 2D | box sai | 960 | 860 | ≈ 0 | ✅ |
| 7 | OC-SORT (OCR) | 2D | điểm lan truyền | 2096 | 2071 | < 2% | ❌ |
| 8 | OC-SORT thêm vào ByteTrack | 2D | điểm lan truyền | 2180 | 2173 | < 2% | ❌ |
| 9 | Optical flow cho QA temporal | 2D | lỗi lọt duyệt lô | 1632 | 2082 | +0.10 s / keyframe | ❌ |
| 10 | Tính lại score theo sweep | 2D | box phải xem tay | 1707 | 648 | +0.003 s / keyframe | tuỳ chọn |
| 11 | Gộp 4 mô hình LiDAR | 3D | mAP | 0.578 | 0.637 | ×4.1 | ✅ |
| 12 | Tinh chỉnh theo track | 3D | mAP / NDS | 0.637 / 0.690 | 0.668 / 0.713 | +0.005 s | ✅ |
| 13 | Thêm 2 mô hình camera | 3D | mAP | 0.668 | 0.662 | ×3.1 | ❌ |
| 14 | Kiểm tra chéo 3D bằng 2D | 3D | box sửa đúng / làm hỏng (dev) | — | 14 / 39 | — | ❌ |
| 15 | VESPA: box 3D từ mask 2D + LiDAR | 3D | mAP dev | 0.644 | 0.627 | — | ❌ |
| 16 | VESPA: hướng theo chuyển động, cỡ theo lớp | 3D | NDS | 0.713 | 0.712–0.713 | ≈ 0 | ❌ |
| 17 | Lan truyền 3D theo vận tốc mô hình | 3D | đổi ID (held-out 4) | 98 | 36 | ≈ 0 | ✅ |
| 18 | Giữ track 3D 4 keyframe (ý OC-SORT) | 3D | điểm dev / nhãn đúng test | 3731 / 23355 | 3762 / 23731 | ≈ 0 | ✅ |
| 19 | OC-SORT OCR / ORU / OCM cho 3D | 3D | điểm dev | 3762 | 3731–3734 | ≈ 0 | ❌ |

## 8. Thời gian suy luận

Biểu đồ: [Thời gian suy luận](figures.ipynb) (mục 11 trong notebook).

| Bước | GPU laptop (RTX 4050) | CPU |
|---|---|---|
| Detector 2D YOLOE-26-L, mỗi ảnh | 0.076 s | 3.31 s |
| Toàn pipeline 2D, mỗi keyframe (detect keyframe + 4 sweep, chiếu LiDAR, QA) | ~1 s | detect 17.7 s |
| Xử lý sau detector khi đã có cache, mỗi keyframe | 0.044 s | — |
| Detector 3D mặc định (4 mô hình LiDAR + gộp + track), mỗi keyframe | 2.56 s | không chạy được (cần op CUDA) |
| Toàn pipeline 3D, mỗi keyframe (4 mô hình + kiểm chứng 6 camera) | ~4 s | — |
| Lan truyền 2D, mỗi lần (≤ 10 keyframe, khoảng 60 ảnh 12 Hz) | 1.39 s (0.33 s khi không flow) | — |
| Lan truyền 3D | không đáng kể | không đáng kể |

### Thời gian trước / sau mỗi lần đổi model và tối ưu

| # | Thay đổi | Đơn vị (máy) | Trước | Sau | Thời gian | Chất lượng | Dùng? |
|---|---|---|---|---|---|---|---|
| 1 | Detector 2D: YOLO-World → YOLOE-26-L | s / ảnh (CPU) | 3.75 | 3.31 | −12% | mAP50 0.311 → 0.452 | ✅ |
| 2 | Detector 2D: YOLO26 → YOLOE-26-L | s / ảnh (CPU) | 2.14 | 3.31 | +55% | mAP50 0.398 → 0.452, có lớp barrier | ✅ |
| 3 | YOLOE-26-L zero-shot → fine-tune nuImages | s / ảnh (GPU) | 0.076¹ | 0.076 | không đổi | mAP50 0.312 → 0.589 (linear probe 0.366) | ✅ |
| 4 | Detector 3D: PointPillars → CenterPoint voxel | s / keyframe (GPU) | 0.85 | 0.62 | −27% | mAP 0.473 → 0.578 | ✅ |
| 5 | CenterPoint voxel → + tinh chỉnh theo track | s / keyframe (GPU) | 0.62 | 0.63 | +1% | mAP 0.578 → 0.616 | — |
| 6 | CenterPoint voxel → gộp 4 LiDAR | s / keyframe (GPU) | 0.62 | 2.55 | ×4.1 | mAP 0.578 → 0.637 | ✅ |
| 7 | Gộp 4 LiDAR → + tinh chỉnh theo track (mặc định) | s / keyframe (GPU) | 2.55 | 2.56 | +0.4% | mAP 0.637 → 0.668 | ✅ |
| 8 | Mặc định → + 2 mô hình camera | s / keyframe (GPU) | 2.56 | 8.01 | ×3.1 | mAP 0.668 → 0.662 | ❌ |
| 9 | Ngưỡng giữ box 2D 0.30 | — | — | — | không đổi | precision 0.34 → 0.51 | ✅ |
| 10 | Optical flow cho QA temporal | s / keyframe (GPU) | 0.044 | 0.143 | ×3.3 | lỗi lọt duyệt lô 1632 → 2082 | ❌ |
| 11 | Tính lại score theo sweep (mean) | s / keyframe (GPU) | 0.044 | 0.047 | +7% | box xem tay 1707 → 648 | tuỳ chọn |
| 12 | Optical flow khi lan truyền 2D | s / lần lan truyền (GPU) | 0.33 | 1.39 | ×4.2 | nhãn đúng 2866 → 3220, đổi ID 205 → 82 | ✅ |
| 13 | ByteTrack (có flow) | s, dev (cùng máy) | 67.6 | 67.4 | ≈ 0 | box sai 960 → 860 | ✅ |
| 14 | OC-SORT OCR, không ByteTrack | s, scene-0093 (VM)² | 16.2 | 16.5 | +2% | điểm 2096 → 2071 | ❌ |
| 15 | OC-SORT OCR thêm vào ByteTrack | s, scene-0093 (VM)² | 16.4 | 16.2 | ≈ 0 | điểm 2180 → 2173 | ❌ |
| 16 | TTA lật ảnh / ảnh 1600 px | s / ảnh | — | — | chưa đo (ước tính ×2 / ×1.6) | mAP50 dev 0.395 → 0.381 / 0.365 | ❌ |
| 17 | Lan truyền 3D: vận tốc, giữ 4 keyframe, OC-SORT 3D | — | — | — | ≈ 0, không chạy mô hình | đổi ID 98 → 36; nhãn đúng +1.6% | ✅ / ❌ |

Ghi chú:

1. Lần chấm zero-shot trên GPU đọc lại từ cache, nên không có số riêng. Hai bản cùng kiến trúc YOLOE-26-L và cùng 1280
   px; linear probe chỉ học lại đầu phân lớp, bản toàn mạng đo được 0.069 s / ảnh (957 ảnh / 66.5 s).
2. Đo trên máy ảo CPU với optical flow đã tính sẵn, để so riêng phần ghép của tracker.

GPU mạnh hơn không làm accuracy cao hơn khi dùng cùng một mô hình, vì cấu hình CPU và GPU giống hệt nhau, chỉ khác FP16.
GPU cho phép chạy được những phần tăng accuracy: 4 mô hình LiDAR, fine-tune, ảnh 1280 px.

## 9. Hạn chế và việc tiếp theo

- **Bảng lan truyền 2D và QA Agent 2D** ở các mục trên vẫn là số đo với detection zero-shot / linear probe. Số với model
  fine-tune toàn mạng ở `eval/results/temporal/report.md`, mục 0b (chạy 04/10).
- **Gộp 2D + 3D** đã đo trên 6 camera với detector fine-tune toàn mạng; CAM_BACK_RIGHT yếu nhất sau khi gộp (F1 0.674),
  chưa tìm nguyên nhân. QA Agent 2D và lan truyền 2D vẫn chỉ đo trên CAM_FRONT.
- **QA Agent 2D** đã dò lại cho detector fine-tune toàn mạng (mục 4): box sai lọt duyệt theo lô 3621 → 280, nhưng người
  phải xem 66% số box. Nhóm high (25% số box, 95% sai) chưa có thao tác xoá theo lô trên giao diện.
- **Precision của detector 2D** vẫn quanh 0.48 ở ngưỡng 0.30; construction vehicle và trailer còn yếu (AP50 0.11 / 0.22).
- **P/R/F1 của 3D** đo trên 20/24 scene test. Đây là phần có sẵn bảng nhãn đã lọc trên máy; mAP và NDS vẫn dùng đủ 24
  scene.
- **Ngưỡng score 0.3** dùng chung cho mọi mô hình 3D nên không công bằng với PGD, vốn cho điểm thấp.
- **Kiểm tra temporal của QA** so box bằng IoU mà không bù chuyển động, nên báo sai với vật ở gần đang chạy nhanh qua ảnh
  (`docs/eval-evidence.md`, TC-11).
- **Chưa làm:** BEVFusion (công bố 68.6 mAP, cần biên dịch op CUDA), TTA lật
  trục cho mô hình LiDAR, ByteTrack cho 3D.

## Nguồn số liệu

| Phần | File |
|---|---|
| 3D mAP / NDS / sai số / AP từng lớp | `eval/results/det3d/heldout24.json`, `det3d_heldout.md` |
| 3D P / R / F1 | `eval/results/det3d/pr_test20.json` |
| 3D thời gian, VRAM | `eval/results/det3d/*/run.json` |
| QA 3D | `eval/results/det3d/*/verify_eval.json` |
| Detector 2D | `eval/results/det2d_finetune.json`, `det2d_variants.json`, `compare/*/autolabel2d_eval.json`, `compare/speed.json` |
| Pipeline và QA 2D | `eval/results/temporal/gpu_heldout20.json` |
| Lan truyền 2D | `eval/results/tracker_test20.json`, `ocsort.json`, `temporal/bytetrack.json` |
| Lan truyền 3D | `eval/results/ocsort.json`, `propagation3d.md` |
| Bảng các phương pháp | `eval/results/bang-so-sanh.md` |
| Test case thủ công | `docs/eval-evidence.md` |
