# PRD — AutoLabel 3D: Tự động gán nhãn Ảnh & LiDAR (human-in-the-loop)

2026-09-19 · @Someone · Nhóm 4 người · Thời lượng 6 tuần

## Thông tin tài liệu

| Mục | Nội dung |
| :---- | :---- |
| Tên đề tài | Công cụ tự động gán nhãn ảnh & LiDAR (2D/3D box, segmentation) với human-in-the-loop |
| Tên sản phẩm | AutoLabel 3D |
| Lĩnh vực | Perception cho xe tự hành — công cụ gán nhãn dữ liệu |
| Nhóm thực hiện | 4 thành viên (ML 2D, ML 3D, Backend, Frontend) |
| Thời lượng | 6 tuần, chia hai giai đoạn: mức cơ bản (tuần 1–4) và mức nâng cao (tuần 5–6) |
| Hạ tầng | GPU miễn phí trên Colab/Kaggle; backend và frontend chạy cục bộ qua Docker |
| Trạng thái | Bản nháp đang hoàn thiện — ngưỡng chất lượng chưa chốt |
| Phạm vi dữ liệu | Subset nuScenes mini / KITTI, mở rộng dần trong kỳ |

## Mục lục

1. [Tóm tắt](#myn0hpr8e35.125)  
2. [Bối cảnh và vấn đề](#myn0hpr8e35.1246)  
3. [Mục tiêu và chỉ số thành công](#myn0hpr8e35.2322)  
4. [Phạm vi](#myn0hpr8e35.3711)  
5. [Sản phẩm cần nộp](#myn0hpr8e35.19158)  
6. [Người dùng và vai trò](#myn0hpr8e35.4939)  
7. [Chân dung người dùng và user stories](#myn0hpr8e35.32138)  
8. [Luồng người dùng chính](#myn0hpr8e35.5971)  
9. [Yêu cầu chức năng](#myn0hpr8e35.6940)  
10. [Tiêu chí chấp nhận chi tiết](#myn0hpr8e35.34279)  
11. [Tình huống biên và cách xử lý](#myn0hpr8e35.36778)  
12. [Yêu cầu phi chức năng](#myn0hpr8e35.9588)  
13. [Kiến trúc và công nghệ](#myn0hpr8e35.10642)  
14. [Đặc tả API và schema](#myn0hpr8e35.38871)  
15. [Mô hình dữ liệu và định dạng nhãn](#myn0hpr8e35.12219)  
16. [Công thức đo và hàm tính điểm](#myn0hpr8e35.41232)  
17. [Kế hoạch đánh giá](#myn0hpr8e35.13548)  
18. [Kịch bản demo và danh sách màn hình](#myn0hpr8e35.43156)  
19. [Kế hoạch 6 tuần cho 4 người](#myn0hpr8e35.14864)  
20. [Rủi ro và giả định](#myn0hpr8e35.16269)  
21. [Hướng mở rộng sau kỳ này](#myn0hpr8e35.29338)  
22. [Checklist nghiệm thu](#myn0hpr8e35.45662)

## Tóm tắt

AutoLabel 3D là công cụ web sinh nhãn sơ bộ cho ảnh và point cloud bằng mô hình pretrained, rồi đưa nhãn đó vào giao diện để annotator và reviewer sửa, xác nhận trước khi nhãn được dùng cho tập train. Trong 6 tuần, nhóm làm cả hai mức của đề tài. Mức cơ bản là công cụ chạy được trên một frame đồng bộ camera \+ LiDAR: tự sinh 2D box/segmentation và 3D box, cho sửa ở cả hai không gian với chiếu 2D↔3D, có luồng duyệt hai vai trò, xuất nhãn theo chuẩn nuScenes/KITTI và báo cáo mAP/IoU so với ground truth. Mức nâng cao mở rộng sang cả một sequence: chạy batch, tracking object qua frame, xếp hạng frame khó và thống kê năng suất.

Giá trị đo được của sản phẩm là thời gian gán nhãn mỗi frame giảm so với làm thủ công, với điều kiện chất lượng nhãn cuối không giảm. Vì vậy PRD này coi hai số liệu — thời gian mỗi frame và tỷ lệ nhãn tự động phải sửa — là tiêu chí nghiệm thu chính, không phải số lượng tính năng.

Hệ thống chỉ xử lý dữ liệu đã ghi, không điều khiển phương tiện và không có thành phần thời gian thực trên xe. Mọi nhãn tự động đều bắt buộc qua người duyệt trước khi xuất.

**Điều kiện triển khai nhóm đã chốt:** làm cả mức cơ bản lẫn mức nâng cao; chạy inference trên GPU miễn phí của Colab/Kaggle; dataset và danh sách lớp phương tiện mở rộng dần trong suốt dự án; ngưỡng chất lượng chưa chốt, sẽ đặt sau khi có baseline.

## Bối cảnh và vấn đề

Gán nhãn dữ liệu perception là khâu tốn kém nhất trong vòng đời một mô hình AV. Một frame đầy đủ gồm 2D box và segmentation trên ảnh, cộng 3D box trên point cloud, và phải nhất quán giữa hai không gian đó.

Ba vấn đề cụ thể mà sản phẩm nhắm tới:

1. **Chi phí thời gian.** Vẽ thủ công 3D box trên point cloud chậm hơn nhiều lần so với 2D box, vì annotator phải xoay góc nhìn và ước lượng kích thước, hướng (yaw) từ tập điểm thưa.  
2. **Không đồng nhất giữa người gán nhãn.** Cùng một vật thể, hai annotator cho ra box lệch nhau về biên và nhãn lớp, kéo chất lượng tập train xuống mà không ai đo được độ lệch đó.  
3. **Thiếu đồng bộ 2D–3D.** Nhãn ảnh và nhãn LiDAR thường làm rời ở hai công cụ khác nhau, nên một vật thể có thể có ID hay lớp khác nhau giữa hai modal.

Hướng giải quyết là đảo ngược vai trò của người: thay vì vẽ từ đầu, người chỉ sửa và duyệt đề xuất của mô hình pretrained. Đây là human-in-the-loop bắt buộc chứ không phải tùy chọn: mô hình pretrained không đủ tin cậy để nhãn đi thẳng vào tập train, còn người thì không cần làm lại phần mà mô hình đã làm đúng.

## Mục tiêu và chỉ số thành công

Mục tiêu sản phẩm: giảm thời gian gán nhãn một frame camera \+ LiDAR mà không làm giảm chất lượng nhãn cuối cùng, và làm cho chất lượng đó trở nên đo được.

| \# | Chỉ số | Cách đo | Ngưỡng tham khảo |
| :---- | :---- | :---- | :---- |
| M1 | Thời gian gán nhãn mỗi frame | Đồng hồ trong app, từ lúc mở frame tới lúc approve | Giảm so với baseline thủ công của chính nhóm |
| M2 | mAP của nhãn tự động 3D | So với ground truth của dataset, IoU 3D ≥ 0.5 | Báo cáo theo từng lớp |
| M3 | mAP của nhãn tự động 2D | So với ground truth, IoU 2D ≥ 0.5 | Báo cáo theo từng lớp |
| M4 | Tỷ lệ nhãn phải sửa | Số object bị người chỉnh hoặc xóa / tổng object tự sinh | Theo dõi xu hướng giảm dần |
| M5 | Chất lượng sau khi người duyệt | IoU trung bình của nhãn đã approve so với ground truth | Không thấp hơn nhánh làm thủ công |
| M6 | Độ trễ auto-label một frame | Từ lúc bấm nút tới lúc nhãn hiển thị | Đo và ghi nhận trên GPU Colab/Kaggle |
| M7 | Ẩn danh | Tỷ lệ khuôn mặt / biển số được làm mờ trong ảnh hiển thị | Đo trên tập kiểm thử thủ công |

Nhóm chưa chốt ngưỡng đạt cụ thể cho các chỉ số này, nên cột cuối mới chỉ là cách đọc số liệu. Cách đặt ngưỡng hợp lý nhất là đợi đủ hai số gốc: baseline thời gian gán nhãn thủ công (tuần 1\) và mAP của mô hình pretrained trên dataset đã chọn (tuần 2), rồi chốt ngưỡng cuối tuần 2 và ghi ngược vào bảng này. Đặt ngưỡng trước khi có hai số đó chỉ là đoán.

M1 phụ thuộc một baseline đo trước: tuần 1, mỗi thành viên gán nhãn thủ công 5 frame bằng CVAT để lấy số gốc. Không có số này thì không có gì để so.

**Non-goals.** Huấn luyện hay fine-tune mô hình mới; điều khiển phương tiện hay bất kỳ thành phần thời gian thực nào trên xe; đạt SOTA về độ chính xác detection; thay thế hoàn toàn annotator; hỗ trợ nhiều dataset ngoài nuScenes và KITTI.

## Phạm vi

**Giai đoạn 1 — mức cơ bản (tuần 1–4).**

- Nạp một frame gồm ảnh camera \+ point cloud LiDAR \+ file calibration, từ subset nuScenes mini hoặc KITTI.  
- Tự sinh 2D box và 2D segmentation mask trên ảnh.  
- Tự sinh 3D box trên point cloud.  
- Chiếu 2D↔3D: chọn object ở một bên thì bên kia sáng theo, và ghi nhận cặp object tương ứng.  
- Editor sửa nhãn ở cả hai không gian: thêm, xóa, chỉnh biên, đổi lớp, chỉnh vị trí–kích thước–yaw của 3D box.  
- Hai vai trò annotator và reviewer, với trạng thái duyệt và luồng từ chối.  
- Xuất nhãn đã duyệt theo định dạng nuScenes hoặc KITTI.  
- Làm mờ khuôn mặt và biển số trong ảnh trước khi hiển thị.  
- Trang báo cáo mAP/IoU của nhãn tự động so với ground truth, và thời gian gán nhãn mỗi frame.

**Giai đoạn 2 — mức nâng cao (tuần 5–6).** Nhóm quyết định làm đủ bốn mục, nên chúng nằm trong phạm vi chứ không phải roadmap.

- Pipeline auto-label chạy batch cả sequence, có hàng đợi và tiến độ, giữ đồng bộ 2D–3D qua cả loạt frame.  
- Tracking object qua frame: `object_id` ổn định giữa các frame và nội suy box giữa hai frame người đã sửa.  
- Thống kê năng suất và độ chính xác theo người và theo phiên làm việc.  
- Active learning: xếp hạng và ưu tiên frame khó trong hàng đợi.  
- Tối ưu throughput inference: batch, half precision, cache kết quả; đo frame/giờ trên một phiên GPU.

**Ngoài phạm vi kỳ này.**

- Huấn luyện lại mô hình từ nhãn đã duyệt; active learning chỉ dừng ở xếp hạng frame khó.  
- Đa người dùng cùng sửa một frame theo thời gian thực.  
- 3D semantic segmentation trên point cloud; chỉ làm 3D box.  
- Triển khai production, SSO, phân quyền chi tiết, audit log đầy đủ.

Ranh giới quan trọng nhất: hệ thống chỉ xử lý dữ liệu đã ghi. Không có module nào gắn vào phương tiện đang chạy, không có đầu ra nào đi vào hệ điều khiển.

## Sản phẩm cần nộp

Yêu cầu nộp của đề tài chia hai mức. Bảng dưới ánh xạ từng gạch đầu dòng sang yêu cầu chức năng tương ứng, để không sót mục nào khi nghiệm thu.

**Mức cơ bản — bắt buộc, là toàn bộ MVP 6 tuần.**

| Yêu cầu đề bài | Yêu cầu chức năng | Tuần hoàn thành |
| :---- | :---- | :---- |
| Web tool nạp 1 frame ảnh \+ LiDAR | FR-01, FR-02 | 2–3 |
| Tự sinh box và segmentation sơ bộ | FR-04, FR-05 | 2 |
| Giao diện sửa nhãn 2D và 3D | FR-09 → FR-14 | 3–4 |
| Xuất nhãn chuẩn nuScenes/KITTI | FR-17, FR-18 | 5 |
| ≥ 2 vai trò (annotator, reviewer) | FR-15, FR-16, FR-21 | 5 |
| Báo cáo IoU/mAP so ground truth | FR-19 | 5–6 |

Hai điểm điều chỉnh so với bản PRD ban đầu, vì đề bài đã nói rõ: segmentation là bắt buộc chứ không phải tùy chọn, nên FR-04 giữ mức M; và FR-10 (brush sửa mask) vẫn có thể cắt nếu thiếu thời gian, vì đề bài chỉ yêu cầu *sinh* segmentation sơ bộ, không bắt phải sửa mask bằng tay.

**Mức nâng cao — nhóm làm đủ bốn mục, tập trung ở tuần 5–6.**

| Yêu cầu đề bài | Cách triển khai | Tuần |
| :---- | :---- | :---- |
| Auto-label batch cả sequence, đồng bộ 2D–3D | Mở rộng job queue thành job theo sequence, có tiến độ và checkpoint từng frame | 5 |
| Tracking object qua frame | Ghép cặp box giữa hai frame liền nhau theo IoU 3D, giữ `object_id`, nội suy box giữa hai frame người đã sửa | 5–6 |
| Thống kê năng suất và độ chính xác | Mở rộng trang báo cáo: tách số liệu theo người và theo phiên làm việc | 6 |
| Active learning ưu tiên frame khó | Điểm độ khó \= confidence thấp \+ số object nhiều \+ bất đồng 2D–3D; sắp xếp hàng đợi theo điểm này | 6 |
| Tối ưu throughput inference | Batch inference, half precision, cache kết quả theo frame; đo frame/giờ mỗi phiên GPU | 6 |

Bốn mục nâng cao trong hai tuần với bốn người là lịch căng, nên thứ tự trên không phải ngẫu nhiên: batch và tracking làm trước vì hai mục còn lại dựa trên chúng — active learning cần confidence của cả loạt frame, tối ưu throughput chỉ đo được khi đã chạy batch. Nếu phải cắt, cắt theo chiều ngược lại: tối ưu throughput trước, rồi active learning.

## Người dùng và vai trò

Hệ thống có ba vai trò, trong đó hai vai trò đầu bắt buộc phải có trong MVP.

| Vai trò | Mục tiêu khi dùng | Được làm | Không được làm |
| :---- | :---- | :---- | :---- |
| Annotator | Sửa nhãn tự động cho đúng, nhanh nhất có thể | Chạy auto-label, sửa/thêm/xóa nhãn 2D và 3D, gửi frame đi duyệt | Approve frame của chính mình, xuất dữ liệu |
| Reviewer | Đảm bảo nhãn đủ chất lượng trước khi vào tập train | Xem nhãn, sửa trực tiếp, approve hoặc reject kèm lý do, xuất dữ liệu | — |
| ML Engineer | Theo dõi chất lượng mô hình và năng suất gán nhãn | Xem trang báo cáo mAP/IoU, chọn checkpoint mô hình, xuất dữ liệu | Sửa nhãn đã approve mà không đổi trạng thái frame |

Quy tắc human-in-the-loop, áp dụng ở tầng backend chứ không chỉ ở giao diện:

- Nhãn do mô hình sinh ra luôn mang trạng thái `auto` và không bao giờ được xuất trực tiếp.  
- API export chỉ trả về các frame ở trạng thái `approved`.  
- Mỗi frame `approved` phải ghi lại ai duyệt và duyệt lúc nào.  
- Người duyệt phải khác người gán nhãn. Trong đề tài môn học, các thành viên đổi vai cho nhau giữa các frame.

## Chân dung người dùng và user stories

**Persona 1 — Annotator.** Sinh viên hoặc nhân viên gán nhãn, làm việc theo ca vài giờ liên tục trên cùng một màn hình. Quen 2D box, lúng túng với point cloud. Điều họ sợ nhất là mất công đã làm vì trình duyệt treo hoặc bấm nhầm. Đo thành công bằng số frame xong mỗi giờ.

**Persona 2 — Reviewer.** Thành viên nắm tiêu chuẩn nhãn, duyệt frame của người khác. Cần nhìn nhanh chỗ nào đáng ngờ thay vì soát lại từng object, và cần lý do từ chối đến đúng người gán nhãn.

**Persona 3 — ML Engineer.** Người dùng tập nhãn để train. Quan tâm chất lượng và truy vết: nhãn này do mô hình nào sinh, ai duyệt, phiên bản dataset nào.

| Mã | Vai trò | Mong muốn | Để mà |
| :---- | :---- | :---- | :---- |
| US-01 | Annotator | Mở một frame và thấy ngay nhãn sơ bộ trên cả ảnh và point cloud | Tôi sửa thay vì vẽ lại từ đầu |
| US-02 | Annotator | Điều chỉnh ngưỡng confidence để ẩn nhãn rác | Khung hình không bị phủ kín bởi box sai |
| US-03 | Annotator | Chọn một object ở khung ảnh và thấy nó sáng lên trong point cloud | Tôi biết hai bên đang nói về cùng một vật thể |
| US-04 | Annotator | Chỉnh vị trí, kích thước và hướng của 3D box bằng chuột hoặc nhập số | Tôi sửa được box lệch mà không phải xoá đi vẽ lại |
| US-05 | Annotator | Được lưu nháp tự động | Mất mạng hay lỡ tải lại trang không mất công đã làm |
| US-06 | Annotator | Gửi frame đi duyệt và thấy lý do khi bị trả về | Tôi sửa đúng chỗ reviewer chê |
| US-07 | Reviewer | Xem frame đang chờ duyệt kèm dấu vết chỉnh sửa của annotator | Tôi tập trung vào phần người ta đã động vào |
| US-08 | Reviewer | Approve hoặc reject kèm lý do | Nhãn kém không lọt vào tập train |
| US-09 | Reviewer | Xuất nhãn đã duyệt theo chuẩn nuScenes/KITTI | Nhóm dùng được ngay bằng devkit có sẵn |
| US-10 | ML Engineer | Xem mAP/IoU của nhãn tự động so với ground truth | Tôi biết mô hình pretrained đang giúp được đến đâu |
| US-11 | ML Engineer | Xem thời gian mỗi frame và tỷ lệ nhãn phải sửa | Tôi chứng minh được công cụ tiết kiệm thời gian thật |
| US-12 | ML Engineer | Chạy auto-label cả một sequence rồi quay lại xem kết quả | Tôi không phải ngồi bấm từng frame |
| US-13 | ML Engineer | Hàng đợi đưa frame khó lên trước | Công sức của người đổ vào chỗ mô hình yếu nhất |

## Luồng người dùng chính

flowchart TD

&nbsp;&nbsp;A\[Nạp frame:\<br/\>ảnh \+ LiDAR \+ calib\] \--\> B\[Ẩn danh\<br/\>mặt & biển số\]

&nbsp;&nbsp;B \--\> C\[Auto-label\<br/\>2D \+ 3D\]

&nbsp;&nbsp;C \--\> D\[Annotator sửa\<br/\>2D và 3D\]

&nbsp;&nbsp;D \--\> E{Reviewer\<br/\>duyệt}

&nbsp;&nbsp;E \--\>|Reject \+ lý do| D

&nbsp;&nbsp;E \--\>|Approve| F\[Nhãn approved\]

&nbsp;&nbsp;F \--\> G\[Export\<br/\>nuScenes / KITTI\]

&nbsp;&nbsp;F \--\> H\[Báo cáo\<br/\>mAP / IoU / thời gian\]

Annotator mở một frame, bấm chạy auto-label, chờ nhãn sơ bộ hiện lên ở cả khung ảnh và khung point cloud, sửa những chỗ sai, rồi gửi đi duyệt. Reviewer xem lại, sửa nếu cần, và approve hoặc reject kèm lý do; frame bị reject quay về hàng đợi của annotator với ghi chú của reviewer.

Ba điểm cần giữ đúng trong luồng này:

- Ẩn danh chạy trước khi ảnh được trả về trình duyệt, không phải làm mờ ở phía client.  
- Đồng hồ đo thời gian chạy từ lúc mở frame tới lúc approve, và dừng khi tab không được focus quá 60 giây.  
- Mỗi lần chạy lại auto-label trên frame đã có người sửa phải hỏi xác nhận, vì nó ghi đè nhãn đang có.

## Yêu cầu chức năng

Ưu tiên theo MoSCoW: M \= bắt buộc, S \= nên có, C \= có thể có nếu còn thời gian. FR-01 đến FR-22 thuộc mức cơ bản, FR-23 đến FR-27 thuộc mức nâng cao.

| ID | Yêu cầu | Ưu tiên | Tiêu chí chấp nhận |
| :---- | :---- | :---- | :---- |
| FR-01 | Nạp một frame gồm ảnh, file point cloud (.pcd/.bin) và calibration | M | Upload sai định dạng báo lỗi rõ; frame hợp lệ hiển thị được cả hai khung trong 5 giây |
| FR-02 | Chọn frame từ subset nuScenes mini / KITTI đã nạp sẵn | M | Danh sách frame có trạng thái và người đang xử lý |
| FR-03 | Ẩn danh khuôn mặt và biển số trên ảnh trước khi hiển thị | M | Ảnh trả về client đã bị làm mờ; đạt M7 |
| FR-04 | Auto-label 2D: box \+ segmentation mask trên ảnh | M | Trả về danh sách object có lớp, box, mask, confidence |
| FR-05 | Auto-label 3D: box trên point cloud | M | Trả về box dạng (x, y, z, w, l, h, yaw), lớp, confidence |
| FR-06 | Ngưỡng confidence điều chỉnh được để lọc nhãn tự động | S | Thanh trượt 0–1, số object hiển thị đổi ngay |
| FR-07 | Chiếu 3D box xuống ảnh và gợi ý ghép cặp với 2D box | M | Cặp có IoU chiếu ≥ 0.5 được ghép tự động; người sửa được cặp ghép |
| FR-08 | Chọn object ở một khung thì khung kia highlight object tương ứng | M | Hai chiều, độ trễ dưới 200 ms |
| FR-09 | Editor 2D: thêm, xóa, kéo biên box, đổi lớp | M | Thao tác bằng chuột; có undo/redo |
| FR-10 | Editor 2D cho mask: tô thêm, xóa bớt bằng brush | S | Brush đổi kích thước được |
| FR-11 | Editor 3D: xem point cloud, xoay, zoom, chọn box | M | Hiển thị mượt ở mức \~100k điểm |
| FR-12 | Editor 3D: chỉnh vị trí, kích thước và yaw của box | M | Có cả gizmo kéo và ô nhập số; có chế độ nhìn từ trên xuống (BEV) |
| FR-13 | Phím tắt cho các thao tác hay dùng | S | Ít nhất: chuyển object, xóa, đổi lớp, lưu, approve |
| FR-14 | Lưu nháp tự động | M | Mất kết nối hoặc tải lại trang không mất quá 30 giây công việc |
| FR-15 | Gửi duyệt, approve, reject kèm lý do | M | Trạng thái frame chuyển đúng; reject bắt buộc nhập lý do |
| FR-16 | Ghi lịch sử: ai sửa gì, lúc nào, nhãn nào do máy sinh và nhãn nào do người sửa | M | Mỗi annotation có trường nguồn và dấu vết chỉnh sửa |
| FR-17 | Export nhãn đã approve theo định dạng nuScenes hoặc KITTI | M | File xuất đọc được bằng devkit tương ứng mà không lỗi |
| FR-18 | Export chặn frame chưa duyệt | M | Frame `auto` hoặc `in_review` không xuất hiện trong file xuất |
| FR-19 | Trang báo cáo: mAP/IoU nhãn tự động vs ground truth, tỷ lệ nhãn phải sửa, thời gian mỗi frame | M | Số liệu cập nhật sau mỗi frame approve; xuất CSV |
| FR-20 | Quản lý phiên bản nhãn bằng DVC | S | Mỗi lần export tạo một phiên bản truy lại được |
| FR-21 | Đăng nhập và phân vai annotator/reviewer | M | Người dùng chỉ thấy hành động thuộc vai của mình |
| FR-22 | So sánh nhãn của hai annotator trên cùng frame | C | Báo cáo IoU giữa hai người (inter-annotator agreement) |
| FR-23 | Auto-label batch cả một sequence, có hàng đợi và tiến độ | M | Job chạy tiếp được sau khi phiên GPU bị ngắt; tiến độ hiển thị theo frame |
| FR-24 | Tracking: giữ object\_id ổn định qua các frame trong sequence | M | Cùng một xe giữ nguyên id qua ít nhất 10 frame liên tiếp; người sửa được liên kết id |
| FR-25 | Nội suy box giữa hai frame người đã sửa | S | Sửa frame đầu và cuối, các frame giữa được điền và đánh dấu là nội suy |
| FR-26 | Active learning: xếp hạng frame theo độ khó và ưu tiên trong hàng đợi | M | Hàng đợi sắp theo điểm độ khó; xem được lý do một frame bị xếp khó |
| FR-27 | Thống kê năng suất theo người và theo phiên, kèm throughput inference | M | Báo cáo có frame/giờ mỗi người và frame/giờ inference mỗi phiên GPU |

## Tiêu chí chấp nhận chi tiết

Viết theo cấu trúc Given – When – Then cho các yêu cầu mức M. Mỗi kịch bản phải chạy được tay trước buổi demo.

### FR-04, FR-05 — Auto-label 2D và 3D

Scenario: Sinh nhãn sơ bộ cho một frame

&nbsp;&nbsp;Given một frame có đủ ảnh, point cloud và calibration

&nbsp;&nbsp;And phiên GPU Colab/Kaggle đang kết nối

&nbsp;&nbsp;When annotator bấm "Chạy auto-label"

&nbsp;&nbsp;Then hệ thống trả về danh sách object có lớp, hình học và confidence cho cả 2D và 3D

&nbsp;&nbsp;And mỗi object được đánh dấu nguồn là \`model\`

&nbsp;&nbsp;And hệ thống ghi lại tên model, phiên bản checkpoint và ngưỡng confidence đã dùng

### FR-07, FR-08 — Đồng bộ 2D–3D

Scenario: Chọn object ở một khung thì khung kia sáng theo

&nbsp;&nbsp;Given một frame đã có nhãn 2D và 3D, các cặp đã được ghép

&nbsp;&nbsp;When annotator chọn một 3D box trong khung point cloud

&nbsp;&nbsp;Then 2D box tương ứng trên ảnh được highlight trong dưới 200 ms

&nbsp;&nbsp;And khi chọn ngược lại từ khung ảnh thì 3D box tương ứng cũng sáng

&nbsp;

Scenario: Object chưa được ghép cặp

&nbsp;&nbsp;Given một 3D box không tìm được 2D box nào có IoU chiếu ≥ 0.5

&nbsp;&nbsp;When annotator chọn 3D box đó

&nbsp;&nbsp;Then hệ thống báo object chưa có cặp và cho phép ghép thủ công

### FR-15, FR-18 — Duyệt và chặn xuất nhãn chưa duyệt

Scenario: Frame bị từ chối quay về annotator

&nbsp;&nbsp;Given một frame ở trạng thái \`in\_review\`

&nbsp;&nbsp;When reviewer bấm "Reject" và nhập lý do

&nbsp;&nbsp;Then frame chuyển về \`rejected\` và hiện trong hàng đợi của annotator kèm lý do

&nbsp;&nbsp;And hệ thống không cho bấm Reject khi lý do để trống

&nbsp;

Scenario: Export chỉ lấy nhãn đã duyệt

&nbsp;&nbsp;Given tập dữ liệu có frame ở cả bốn trạng thái auto, editing, in\_review và approved

&nbsp;&nbsp;When người dùng gọi chức năng export

&nbsp;&nbsp;Then file xuất chỉ chứa frame \`approved\`

&nbsp;&nbsp;And file đọc được bằng devkit nuScenes hoặc KITTI mà không lỗi

### FR-03 — Ẩn danh trước khi hiển thị

Scenario: Ảnh rời server đã ẩn danh

&nbsp;&nbsp;Given một frame có khuôn mặt người đi đường và biển số xe trong ảnh

&nbsp;&nbsp;When client yêu cầu ảnh của frame đó

&nbsp;&nbsp;Then ảnh trả về đã được làm mờ ở phía server

&nbsp;&nbsp;And không có endpoint nào trả về ảnh gốc chưa làm mờ

### FR-23, FR-24 — Batch cả sequence và tracking

Scenario: Phiên GPU bị ngắt giữa job batch

&nbsp;&nbsp;Given một job auto-label đang chạy trên sequence 50 frame và đã xong 20 frame

&nbsp;&nbsp;When phiên Colab/Kaggle bị ngắt

&nbsp;&nbsp;Then 20 frame đã xong vẫn giữ nguyên nhãn

&nbsp;&nbsp;And khi mở phiên GPU mới, job chạy tiếp từ frame 21 chứ không làm lại từ đầu

&nbsp;

Scenario: Giữ object\_id qua các frame

&nbsp;&nbsp;Given một sequence đã chạy auto-label và tracking

&nbsp;&nbsp;When người dùng xem cùng một chiếc xe ở frame 10 và frame 20

&nbsp;&nbsp;Then hai box mang cùng một \`object\_id\`

## Tình huống biên và cách xử lý

| Mã | Tình huống | Hậu quả nếu bỏ qua | Cách xử lý |
| :---- | :---- | :---- | :---- |
| EC-01 | Không có phiên GPU nào đang mở | Annotator bấm auto-label rồi chờ vô vọng | Nút auto-label bị vô hiệu hóa kèm thông báo; giao diện sửa nhãn vẫn dùng được với nhãn đã sinh trước đó |
| EC-02 | Phiên GPU bị ngắt giữa job batch | Mất hàng chục phút tính toán | Checkpoint sau mỗi frame; job chạy tiếp từ frame dở khi có phiên mới |
| EC-03 | Mô hình không phát hiện được object nào | Annotator tưởng hệ thống hỏng | Báo rõ "không tìm thấy object nào ở ngưỡng hiện tại", gợi ý hạ ngưỡng, vẫn cho vẽ tay từ đầu |
| EC-04 | Mô hình sinh quá nhiều box rác | Màn hình phủ kín box, sửa lâu hơn vẽ tay | Thanh trượt confidence; nút xoá hàng loạt theo ngưỡng; ghi nhận vào M4 để biết mô hình đang hại nhiều hơn lợi |
| EC-05 | Frame thiếu calibration hoặc calibration sai | Chiếu 2D↔3D lệch hoàn toàn, nhãn hỏng hàng loạt | Kiểm tra khi nạp frame; thiếu calibration thì chặn auto-label 3D và báo lý do, không chạy tiếp |
| EC-06 | Hai người mở cùng một frame | Người này ghi đè công của người kia | Khóa mềm theo frame: người thứ hai vào ở chế độ chỉ xem, kèm tên người đang giữ |
| EC-07 | Chạy lại auto-label trên frame người đã sửa | Mất toàn bộ chỉnh sửa thủ công | Hỏi xác nhận, nói rõ số object sẽ bị ghi đè; giữ lại nhãn nguồn `human` nếu người dùng chọn |
| EC-08 | Thêm lớp mới giữa kỳ | Nhãn cũ đổi lớp hoặc mất khi xuất | Lớp có id riêng trong CSDL; migration có test trên nhãn đã duyệt; lớp mới không đổi nhãn cũ |
| EC-09 | Point cloud quá lớn so với giới hạn trình duyệt | Tab đơ hoặc sập | Giới hạn 200k điểm, lấy mẫu bớt khi vượt; báo cho người dùng rằng đang xem bản lấy mẫu |
| EC-10 | Mất kết nối khi đang sửa | Mất công nửa chừng | Lưu nháp cục bộ và đồng bộ lại khi có mạng; không mất quá 30 giây công việc |
| EC-11 | Reviewer cũng là người gán nhãn frame đó | Human-in-the-loop chỉ còn trên giấy | Backend chặn approve frame do chính mình sửa; nhóm đổi vai giữa các frame |
| EC-12 | Tracking ghép nhầm hai xe khác nhau | Nhãn sai lan ra cả sequence | Cho phép tách `object_id` tại frame bất kỳ; nhãn nội suy được đánh dấu riêng để reviewer soát kỹ hơn |

## Yêu cầu phi chức năng

| Nhóm | Yêu cầu |
| :---- | :---- |
| Hiệu năng | Auto-label một frame trong khoảng chấp nhận được trên GPU Colab/Kaggle, đo và ghi nhận chứ chưa chốt ngưỡng. Tải frame và hiển thị point cloud \~100k điểm ≤ 3 giây. Thao tác sửa box phản hồi dưới 100 ms. |
| Chi phí GPU | GPU là Colab/Kaggle miễn phí, nên model service phải chịu được phiên bị ngắt: load model một lần rồi giữ trong VRAM, ghi checkpoint sau mỗi frame trong job batch, cache kết quả auto-label theo frame để không chạy lại, và có chế độ CPU để demo khi không có phiên GPU nào. |
| Riêng tư | Ảnh chỉ rời server sau khi đã ẩn danh. Ảnh gốc chưa ẩn danh không được cache ở client. |
| Bảo mật | API yêu cầu token; endpoint export chỉ mở cho reviewer và ML engineer. |
| Khả năng tái lập | Mỗi lần auto-label ghi lại tên model, phiên bản checkpoint và ngưỡng confidence đã dùng. |
| Khả năng dùng | Annotator mới làm xong một frame đầy đủ sau ≤ 10 phút hướng dẫn. |
| Tương thích | Chạy trên Chrome và Edge bản mới, màn hình ≥ 1440×900. Không hỗ trợ mobile. |
| Triển khai | Toàn bộ hệ thống dựng được bằng một lệnh `docker compose up`. |

Giới hạn đã biết và chấp nhận trong kỳ này: một người sửa một frame tại một thời điểm (khóa mềm theo frame), không xử lý point cloud quá 200k điểm, và chỉ hỗ trợ một camera cho mỗi frame.

Hệ quả của việc chỉ có GPU Colab/Kaggle: model service không chạy thường trực cạnh API mà là một phiên notebook mở khi cần, nối về hệ thống qua một đường hầm (ngrok hoặc tương đương). Khi không có phiên nào mở, giao diện sửa nhãn vẫn phải dùng được với nhãn đã sinh trước đó; chỉ nút chạy auto-label bị vô hiệu hóa kèm thông báo.

## Kiến trúc và công nghệ

flowchart LR

&nbsp;&nbsp;FE\[Web app\<br/\>Next.js \+ three.js\] \--\> API\[FastAPI\<br/\>REST\]

&nbsp;&nbsp;API \--\> DB\[(PostgreSQL\<br/\>nhãn, frame, user)\]

&nbsp;&nbsp;API \--\> ST\[(Object storage\<br/\>ảnh, point cloud)\]

&nbsp;&nbsp;API \--\> Q\[Job queue\<br/\>Celery \+ Redis\]

&nbsp;&nbsp;Q \--\> MS\[Model service\<br/\>GPU\]

&nbsp;&nbsp;MS \--\> M2D\[YOLO \+ SAM2\<br/\>2D\]

&nbsp;&nbsp;MS \--\> M3D\[MMDetection3D\<br/\>3D\]

&nbsp;&nbsp;API \--\> EV\[Eval & export\<br/\>nuScenes / KITTI \+ DVC\]

Lý do chọn từng mảnh:

- **FastAPI** — cùng ngôn ngữ Python với phần model, đỡ một lớp chuyển đổi.  
- **Job queue** — auto-label mất nhiều giây nên không chạy đồng bộ trong request; client poll trạng thái job.  
- **Model service tách riêng** — giữ model nạp sẵn trong RAM/VRAM, tránh load lại mỗi lần gọi, và cho phép chạy trên máy GPU khác với máy chạy API.  
- **three.js** — render point cloud trên trình duyệt, không cần cài đặt gì phía annotator.  
- **PostgreSQL** — nhãn và trạng thái duyệt cần truy vấn và ràng buộc, không hợp để ở file JSON rời.  
- **DVC** — phiên bản hóa tập nhãn đã xuất, để mỗi lần train biết mình dùng nhãn phiên bản nào.

Về công cụ gán nhãn có sẵn: đề tài nhắc tới CVAT và Label Studio. Nhóm nên chốt sớm giữa hai hướng, vì nó đổi gần như toàn bộ khối lượng frontend:

| Hướng | Ưu | Nhược |
| :---- | :---- | :---- |
| Tự xây giao diện (Next.js \+ three.js) | Kiểm soát hoàn toàn đồng bộ 2D–3D và đo thời gian; là điểm demo của đồ án | Tốn phần lớn công sức 6 tuần |
| Dùng CVAT, chỉ viết phần auto-label | Đỡ code UI, có sẵn luồng review | Khó làm đồng bộ 2D–3D và khó đo số liệu theo ý; đóng góp riêng của nhóm mờ nhạt hơn |

PRD này đang giả định hướng tự xây giao diện, vì FR-07, FR-08 và M1 khó đạt nếu đi đường CVAT.

## Đặc tả API và schema

Ba nhóm endpoint chính giữa frontend, backend và model service. Chốt schema sớm để người làm frontend không phải chờ người làm model.

| Endpoint | Mục đích | Vai trò được gọi |
| :---- | :---- | :---- |
| `GET /frames` | Danh sách frame kèm trạng thái và người đang giữ | Tất cả |
| `GET /frames/{id}` | Ảnh đã ẩn danh, point cloud, calibration, nhãn hiện có | Tất cả |
| `POST /frames/{id}/autolabel` | Tạo job auto-label cho một frame | Annotator |
| `POST /sequences/{id}/autolabel` | Tạo job batch cho cả sequence | ML Engineer |
| `GET /jobs/{id}` | Trạng thái và tiến độ job | Tất cả |
| `PUT /frames/{id}/annotations` | Lưu nhãn sau khi sửa | Annotator, Reviewer |
| `POST /frames/{id}/submit` | Gửi frame đi duyệt | Annotator |
| `POST /frames/{id}/review` | Approve hoặc reject kèm lý do | Reviewer |
| `POST /export` | Xuất nhãn đã duyệt theo định dạng chọn | Reviewer, ML Engineer |
| `GET /metrics` | Số liệu mAP/IoU, thời gian, tỷ lệ nhãn phải sửa | Tất cả |

### Schema nhãn của một frame

{

&nbsp;&nbsp;"frame\_id": "scene0061\_f012",

&nbsp;&nbsp;"status": "editing",

&nbsp;&nbsp;"annotations": \[

&nbsp;&nbsp;&nbsp;&nbsp;{

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"id": "ann\_10482",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object\_id": "obj\_37",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"type": "box3d",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"label": "car",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"geometry": {"x": 12.4, "y": \-3.1, "z": 0.8, "w": 1.9, "l": 4.6, "h": 1.5, "yaw": 1.57},

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"confidence": 0.82,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"source": "model",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"edited": false

&nbsp;&nbsp;&nbsp;&nbsp;},

&nbsp;&nbsp;&nbsp;&nbsp;{

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"id": "ann\_10483",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object\_id": "obj\_37",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"type": "box2d",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"label": "car",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"geometry": {"x1": 420, "y1": 310, "x2": 690, "y2": 468},

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"confidence": 0.91,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"source": "human",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"edited": true

&nbsp;&nbsp;&nbsp;&nbsp;}

&nbsp;&nbsp;\]

}

### Schema job auto-label

{

&nbsp;&nbsp;"job\_id": "job\_221",

&nbsp;&nbsp;"scope": "sequence",

&nbsp;&nbsp;"target": "scene0061",

&nbsp;&nbsp;"status": "running",

&nbsp;&nbsp;"frames\_total": 50,

&nbsp;&nbsp;"frames\_done": 21,

&nbsp;&nbsp;"model": {"2d": "yolo11x-seg", "3d": "bevfusion-nus", "conf\_threshold": 0.3},

&nbsp;&nbsp;"started\_at": "2026-09-19T08:10:00Z",

&nbsp;&nbsp;"resumable\_from": 22

}

### Mã lỗi cần xử lý rõ

| Mã | Khi nào | Frontend làm gì |
| :---- | :---- | :---- |
| `GPU_UNAVAILABLE` | Không có phiên GPU nào đăng ký | Vô hiệu hóa nút auto-label, hiện hướng dẫn mở phiên |
| `MISSING_CALIBRATION` | Frame thiếu file calibration | Chặn auto-label 3D, cho phép làm 2D |
| `FRAME_LOCKED` | Người khác đang giữ frame | Mở ở chế độ chỉ xem, hiện tên người giữ |
| `SELF_REVIEW_FORBIDDEN` | Reviewer duyệt frame do chính mình sửa | Ẩn nút approve, giải thích lý do |
| `NOTHING_TO_EXPORT` | Không có frame nào ở trạng thái approved | Báo số frame đang chờ duyệt |

## Mô hình dữ liệu và định dạng nhãn

Bốn thực thể chính:

| Thực thể | Trường chính |
| :---- | :---- |
| `frame` | id, sample\_token, đường dẫn ảnh, đường dẫn point cloud, calibration, trạng thái (`new` / `auto` / `editing` / `in_review` / `approved` / `rejected`), người đang giữ |
| `annotation` | id, frame\_id, object\_id, loại (`box2d` / `mask2d` / `box3d`), lớp, hình học, confidence, nguồn (`model` / `human`), đã bị sửa hay chưa |
| `review` | id, frame\_id, reviewer\_id, kết quả, lý do, thời điểm |
| `run` | id, frame\_id, tên model, phiên bản checkpoint, ngưỡng confidence, thời gian chạy |

`object_id` là khóa nối 2D và 3D: một chiếc xe có cùng `object_id` ở cả box2d, mask2d và box3d. Đây là thứ làm cho FR-08 chạy được và làm cho nhãn hai modal nhất quán.

Hệ toạ độ và quy ước:

- 3D box lưu trong hệ toạ độ LiDAR sensor, dạng `(x, y, z, w, l, h, yaw)`, `z` là tâm box.  
- Chiếu 3D→2D dùng ma trận `camera_intrinsic` và phép biến đổi LiDAR→camera lấy từ calibration của dataset.  
- Danh sách lớp không cố định: bắt đầu với một tập nhỏ (`car`, `pedestrian`, `cyclist`, `truck`) và bổ sung dần các lớp phương tiện khác trong suốt dự án. Taxonomy vì vậy lưu trong CSDL chứ không hardcode trong mã, bảng ánh xạ lớp nội bộ sang lớp nuScenes/KITTI là file cấu hình sửa được, và thêm lớp mới không được phá nhãn đã gán trước đó.  
- Chuyển sang KITTI phải đổi quy ước: KITTI lưu toạ độ trong hệ camera và `z` ở đáy box, không phải tâm. Hàm chuyển đổi cần unit test riêng, vì đây là chỗ hay sai lặng.

Lưu trữ nội bộ dùng một schema JSON trung gian; nuScenes và KITTI đều là bộ chuyển đổi đầu ra từ schema đó, không phải hai đường lưu song song.

## Công thức đo và hàm tính điểm

Bốn công thức dưới đây quyết định mọi con số trong báo cáo, nên phải viết thành code dùng chung, không để mỗi người tự tính một kiểu.

**IoU 3D.** Với hai box A và B, tính giao trên mặt phẳng BEV rồi nhân với phần chồng theo trục đứng:

\\mathrm{IoU}\_{3D}(A,B) \= \\frac{\\mathrm{Area}\_{BEV}(A \\cap B)\\cdot h\_{\\cap}}{V\_A \+ V\_B \- \\mathrm{Area}\_{BEV}(A \\cap B)\\cdot h\_{\\cap}}

**mAP.** Với mỗi lớp, sắp các dự đoán theo confidence giảm dần, ghép với ground truth theo ngưỡng IoU, rồi lấy diện tích dưới đường precision–recall; mAP là trung bình AP của các lớp:

AP\_c \= \\int\_0^1 p\_c(r)\\,dr, \\qquad mAP \= \\frac{1}{|C|}\\sum\_{c \\in C} AP\_c

Báo cáo ở hai ngưỡng IoU 0.5 và 0.7, tách riêng 2D và 3D, tách theo từng lớp. Vì danh sách lớp mở rộng dần, `|C|` phải lấy theo tập lớp có mặt trong lần đo đó và ghi kèm vào báo cáo.

**Ghép cặp 2D–3D.** Chiếu tám đỉnh của 3D box qua ma trận calibration, lấy hộp bao của các điểm chiếu, rồi ghép với 2D box theo IoU lớn nhất:

\\mathrm{match}(b\_{3D}) \= \\arg\\max\_{b\_{2D}} \\mathrm{IoU}\_{2D}\\big(\\pi(b\_{3D}),\\, b\_{2D}\\big), \\quad \\text{chỉ nhận khi } \\mathrm{IoU}\_{2D} \\ge 0.5

Mỗi 2D box chỉ được ghép với một 3D box; các cặp còn lại để trống cho người ghép tay.

**Điểm độ khó cho active learning.** Một frame càng đáng đưa lên đầu hàng đợi khi mô hình càng thiếu chắc chắn và hai modal càng bất đồng:

D(f) \= \\alpha\\big(1 \- \\overline{conf}(f)\\big) \+ \\beta\\,\\frac{n\_{unmatched}(f)}{n\_{obj}(f)} \+ \\gamma\\,\\frac{n\_{obj}(f)}{n\_{max}}

Trong đó `conf` là confidence trung bình, `n_unmatched` là số object không ghép được cặp 2D–3D, `n_obj` là số object trong frame. Ba trọng số `α`, `β`, `γ` đặt bằng nhau ở lần chạy đầu, chỉnh lại sau khi đối chiếu với thời gian sửa thực tế của từng frame — đây là cách kiểm chứng xem điểm độ khó có đo đúng cái nó muốn đo hay không.

**Tỷ lệ nhãn phải sửa (M4).** Một object tính là phải sửa nếu bị xóa, đổi lớp, hoặc hình học đổi làm IoU với bản gốc của mô hình xuống dưới 0.9.

## Kế hoạch đánh giá

Phần này là thứ biến đồ án từ "đã làm xong công cụ" thành "chứng minh được công cụ có tác dụng". Nên thiết kế trước, chạy ở tuần 5–6.

**Tập dữ liệu.** 60 frame có sẵn ground truth, lấy từ dataset đang dùng tại thời điểm đo. Chia: 30 frame cho nhánh auto-label, 30 frame cho nhánh thủ công, phân bổ ngẫu nhiên và cân bằng độ khó (số object mỗi frame). Vì dataset và danh sách lớp mở rộng dần, mỗi lần đo phải ghi kèm phiên bản dataset và tập lớp đã dùng, nếu không thì hai lần đo không so được với nhau.

**Thí nghiệm A/B.** Mỗi thành viên làm cả hai nhánh, xen kẽ thứ tự để triệt tiêu hiệu ứng quen tay. Ghi lại:

- Thời gian hoàn thành mỗi frame ở từng nhánh → chỉ số M1.  
- IoU của nhãn cuối so với ground truth ở từng nhánh → chỉ số M5, để kiểm chứng rằng nhanh hơn không đồng nghĩa ẩu hơn.

**Đo chất lượng nhãn tự động.** Chạy auto-label trên toàn bộ 60 frame, so với ground truth: mAP ở các ngưỡng IoU 0.5 và 0.7, riêng cho 2D và 3D, báo cáo theo từng lớp. Lớp `pedestrian` gần như chắc chắn kém hơn `car`; báo cáo tách ra thay vì gộp một số duy nhất.

**Tỷ lệ nhãn phải sửa (M4).** Tính từ log chỉnh sửa: một object tính là "phải sửa" nếu bị xóa, đổi lớp, hoặc hình học đổi làm IoU với bản gốc của mô hình xuống dưới 0.9. Cần chốt ngưỡng này trước khi chạy, đừng chọn sau khi thấy kết quả.

**Kiểm thử kỹ thuật.** Unit test cho bộ chuyển đổi toạ độ và bộ xuất định dạng (đọc lại bằng devkit), test tính mAP trên một ví dụ tự tính tay, và một lượt kiểm thử luồng đầu cuối trước buổi demo.

## Kịch bản demo và danh sách màn hình

Mọi tài liệu, wireframe và buổi bảo vệ dùng chung một kịch bản dưới đây, chạy trên một sequence đã chuẩn bị trước.

\[Khởi động\]

&nbsp;&nbsp;Annotator mở frame đầu tiên của sequence. Ảnh đã ẩn danh mặt và biển số; point cloud hiển thị bên cạnh.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Auto-label một frame\]

&nbsp;&nbsp;Bấm chạy auto-label. Sau vài giây, 2D box \+ mask và 3D box hiện ra; đồng hồ đo thời gian bắt đầu chạy.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Sửa và đồng bộ\]

&nbsp;&nbsp;Hạ ngưỡng confidence để bỏ box rác. Chọn một xe ở khung ảnh, 3D box tương ứng sáng lên.

&nbsp;&nbsp;Chỉnh lại yaw của một 3D box lệch, xóa một object thừa.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Duyệt\]

&nbsp;&nbsp;Gửi frame đi duyệt. Reviewer mở lên, thấy dấu vết chỉnh sửa, reject kèm lý do.

&nbsp;&nbsp;Annotator sửa lại, gửi lại, reviewer approve.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Chạy batch cả sequence\]

&nbsp;&nbsp;ML engineer chạy auto-label cho toàn bộ sequence. Thanh tiến độ chạy theo từng frame.

&nbsp;&nbsp;Ngắt phiên GPU giữa chừng, mở lại, job chạy tiếp từ frame dở.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Tracking và nội suy\]

&nbsp;&nbsp;Mở frame 10 và frame 20: cùng một xe giữ nguyên object\_id.

&nbsp;&nbsp;Sửa frame đầu và frame cuối của một đoạn, các frame giữa được nội suy và đánh dấu riêng.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Hàng đợi theo độ khó\]

&nbsp;&nbsp;Mở hàng đợi: frame khó được đẩy lên đầu, xem được lý do vì sao bị xếp khó.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

\[Xuất và báo cáo\]

&nbsp;&nbsp;Xuất nhãn đã duyệt theo chuẩn nuScenes/KITTI, đọc lại bằng devkit ngay trên màn hình.

&nbsp;&nbsp;Mở trang báo cáo: mAP/IoU, thời gian mỗi frame so với baseline thủ công, tỷ lệ nhãn phải sửa, frame/giờ.

### Hai luồng sử dụng bắt buộc

- **Annotator:** mở frame → chạy auto-label → lọc theo confidence → sửa 2D → sửa 3D → kiểm tra đồng bộ hai khung → gửi duyệt.  
- **Reviewer:** mở hàng đợi → chọn frame chờ duyệt → xem dấu vết chỉnh sửa → sửa nếu cần → approve hoặc reject kèm lý do → xuất nhãn.

### Danh sách màn hình

| Mã | Màn hình | Thành phần chính | FR ánh xạ |
| :---- | :---- | :---- | :---- |
| M1 | Danh sách frame / hàng đợi | Bảng frame, trạng thái, người giữ, điểm độ khó, bộ lọc | FR-02, FR-26 |
| M2 | Màn hình gán nhãn | Khung ảnh và khung point cloud cạnh nhau, thanh confidence, danh sách object, nút auto-label | FR-01, FR-04 → FR-08 |
| M3 | Bảng điều khiển 3D box | Ô nhập x, y, z, w, l, h, yaw; nút chuyển chế độ BEV | FR-11, FR-12 |
| M4 | Màn hình duyệt | Nhãn kèm dấu nguồn model/human, nút approve và reject, ô lý do | FR-15, FR-16 |
| M5 | Job batch | Tiến độ theo frame, trạng thái phiên GPU, nút chạy tiếp | FR-23, FR-25 |
| M6 | Trang báo cáo | mAP/IoU theo lớp, thời gian mỗi frame, tỷ lệ nhãn phải sửa, frame/giờ theo người | FR-19, FR-27 |
| M7 | Màn hình export | Chọn định dạng, số frame đủ điều kiện, lịch sử các lần xuất | FR-17, FR-18, FR-20 |

## Kế hoạch 6 tuần cho 4 người

Phân vai đề xuất, mỗi người một trục chính nhưng đều tham gia gán nhãn ở tuần đánh giá:

| Vai | Trách nhiệm chính |
| :---- | :---- |
| P1 — ML 2D | YOLO \+ SAM2, pipeline ảnh, ẩn danh mặt/biển số |
| P2 — ML 3D | MMDetection3D/BEVFusion, xử lý point cloud, chiếu 2D↔3D |
| P3 — Backend | FastAPI, CSDL, job queue, trạng thái duyệt, export, DVC |
| P4 — Frontend | Next.js, editor 2D, viewer/editor 3D bằng three.js, trang báo cáo |

&nbsp;

| Tuần | Mục tiêu | Done khi |
| :---- | :---- | :---- |
| 1 | Chốt dataset khởi đầu, dựng khung dự án, dựng phiên GPU Colab/Kaggle, đo baseline thủ công | Docker compose chạy; notebook GPU gọi được từ backend; mỗi người gán thủ công 5 frame và có số thời gian gốc |
| 2 | Auto-label 2D và 3D chạy được, API nạp/đọc frame, script tính mAP | JSON nhãn cho một frame; có số mAP đầu tiên trên tập nhỏ → chốt ngưỡng cho M1–M5 |
| 3 | Giao diện hiển thị ảnh \+ point cloud và nhãn tự động | Mở frame trên web thấy cả hai khung với box vẽ sẵn |
| 4 | Sửa nhãn 2D/3D, đồng bộ chọn object, luồng duyệt hai vai, export, ẩn danh | Sửa → duyệt → xuất được file nuScenes/KITTI đọc bằng devkit; hết mức cơ bản trừ trang báo cáo |
| 5 | Trang báo cáo; bắt đầu mức nâng cao: batch cả sequence \+ tracking | Chạy batch một sequence ≥ 20 frame không cần trông; `object_id` giữ được qua các frame |
| 6 | Nội suy box, active learning, tối ưu throughput, thống kê theo người; thí nghiệm A/B và báo cáo | Có số cho M1–M5; hàng đợi sắp theo độ khó; có số frame/giờ; demo đầu cuối chạy không lỗi |

Lịch này dồn toàn bộ mức cơ bản vào bốn tuần đầu để dành hai tuần cuối cho mức nâng cao, nên mốc kiểm soát quan trọng nhất là cuối tuần 4: nếu luồng sửa → duyệt → xuất chưa chạy thông, mức nâng cao phải cắt bớt chứ không phải mức cơ bản. Hai cách cắt đã chấp nhận trước: bỏ brush sửa mask (FR-10) và so sánh hai annotator (FR-22); sau đó mới tới tối ưu throughput và active learning. Tuần 6 không nhận tính năng nằm ngoài danh sách trên.

## Rủi ro và giả định

| Rủi ro | Mức | Dấu hiệu sớm | Giảm thiểu |
| :---- | :---- | :---- | :---- |
| Làm đủ bốn mục nâng cao trong hai tuần cuối | Cao | Hết tuần 4 mức cơ bản chưa chạy thông đầu cuối | Thứ tự cắt đã chốt trước: throughput, rồi active learning; batch và tracking là hai mục giữ lại đến cùng |
| Phiên Colab/Kaggle bị ngắt giữa job batch | Cao | Job dài quá vài chục phút không xong | Checkpoint sau mỗi frame, job tiếp tục được từ frame dở; chạy trước và cache kết quả cho tập đánh giá |
| Hết hạn mức GPU miễn phí đúng tuần demo | Cao | Colab báo giới hạn trong tuần 5 | Nhiều tài khoản luân phiên; lưu sẵn nhãn đã sinh cho các frame demo để không phụ thuộc GPU lúc trình bày |
| Editor 3D tốn nhiều thời gian hơn dự tính | Cao | Hết tuần 3 chưa xoay được point cloud mượt | Làm BEV 2D trước, 3D tự do sau; chỉnh yaw bằng ô nhập số |
| Cài MMDetection3D/BEVFusion hỏng do xung đột CUDA | Cao | Tuần 2 chưa chạy được inference | Khóa phiên bản torch/CUDA trong notebook mẫu ngay tuần 1; dự phòng PointPillars nhẹ hơn |
| Thêm lớp mới giữa chừng làm hỏng nhãn cũ | Trung bình | Nhãn cũ đổi lớp hoặc biến mất sau khi sửa taxonomy | Taxonomy trong CSDL, lớp có id riêng; migration phải có test trên nhãn đã duyệt |
| mAP 3D quá thấp khiến nhãn tự động vô dụng | Trung bình | mAP rất thấp trên lớp car | Đổi checkpoint pretrained đúng dataset đang dùng; hạ ngưỡng confidence để ưu tiên recall |
| Sai quy ước toạ độ khi chiếu 2D↔3D | Trung bình | Box chiếu lệch hẳn khỏi vật thể | Unit test với ground truth: chiếu GT 3D box phải trùng GT 2D box |
| Thí nghiệm A/B bị dồn vào tuần cuối | Trung bình | Chưa chốt giao thức đo ở tuần 4 | Script tính mAP viết từ tuần 2; chia frame cho hai nhánh ngay khi luồng duyệt chạy |

Giả định đang dựa vào, nếu sai thì kế hoạch phải điều chỉnh: có checkpoint pretrained phù hợp cho dataset đã chọn; mức GPU miễn phí của Colab/Kaggle đủ cho việc chạy batch ở tuần 5–6; cả 4 thành viên dành được thời gian gán nhãn trong tuần đánh giá; việc mở rộng dataset và danh sách lớp diễn ra từng đợt nhỏ chứ không dồn vào cuối kỳ.

## Hướng mở rộng sau kỳ này

Cả mức cơ bản và mức nâng cao đều nằm trong 6 tuần, nên phần này chỉ ghi lại những hướng đi tiếp nếu đề tài được làm tiếp, không thuộc cam kết kỳ này:

1. **Huấn luyện lại mô hình từ nhãn đã duyệt.** Đóng vòng lặp active learning: nhãn người sửa quay lại làm dữ liệu train, mô hình tốt lên thì tỷ lệ nhãn phải sửa giảm.  
2. **Nhiều camera mỗi frame.** nuScenes có sáu camera; hỗ trợ đủ vòng cho phép kiểm tra chéo 3D box từ nhiều góc nhìn.  
3. **3D semantic segmentation.** Gán nhãn từng điểm trong point cloud, không chỉ 3D box.  
4. **Nhiều người cùng sửa một frame** và hạ tầng GPU riêng thay cho phiên Colab/Kaggle.

## Checklist nghiệm thu

Dùng để tự chấm trước buổi bảo vệ. Một mục chỉ được tích khi có thể đem ra cho người khác xem, không phải khi "gần xong".

**Mức cơ bản**

- [ ] Nạp được một frame ảnh \+ LiDAR \+ calibration và hiển thị cả hai khung  
- [ ] Auto-label sinh được 2D box, segmentation mask và 3D box  
- [ ] Sửa được nhãn ở cả hai không gian, có undo và lưu nháp  
- [ ] Chọn object ở một khung thì khung kia sáng theo  
- [ ] Đủ hai vai trò với luồng approve và reject kèm lý do  
- [ ] Backend chặn approve frame do chính người đó sửa  
- [ ] Export ra nuScenes hoặc KITTI, đọc lại được bằng devkit  
- [ ] Export không chứa frame chưa duyệt  
- [ ] Ảnh hiển thị đã ẩn danh mặt và biển số từ phía server  
- [ ] Trang báo cáo hiện mAP/IoU so ground truth, thời gian mỗi frame và tỷ lệ nhãn phải sửa

**Mức nâng cao**

- [ ] Chạy batch được cả một sequence, có tiến độ theo frame  
- [ ] Job chạy tiếp được sau khi phiên GPU bị ngắt  
- [ ] `object_id` giữ ổn định qua ít nhất 10 frame liên tiếp  
- [ ] Nội suy box giữa hai frame người đã sửa, nhãn nội suy được đánh dấu riêng  
- [ ] Hàng đợi sắp theo điểm độ khó và xem được lý do  
- [ ] Báo cáo có số frame/giờ theo người và frame/giờ inference mỗi phiên GPU

**Tài liệu và chứng cứ**

- [ ] Có số baseline thủ công đo ở tuần 1  
- [ ] Ngưỡng cho M1–M5 đã chốt và ghi ngược vào bảng chỉ số  
- [ ] Có kết quả thí nghiệm A/B giữa nhánh auto-label và nhánh thủ công  
- [ ] Mỗi lần đo ghi kèm phiên bản dataset, tập lớp và checkpoint mô hình  
- [ ] Unit test cho bộ chuyển đổi toạ độ và bộ xuất định dạng  
- [ ] Toàn hệ thống dựng được bằng một lệnh `docker compose up`  
- [ ] Kịch bản demo chạy trọn một lượt không lỗi trước ngày bảo vệ

&nbsp;