# Hoàn thiện notebook với bộ ResLinCNN_Lite trên cache v2

File thực hiện: `D:/IOT/CuoiKy/PPG_CS_End_to_End_EDA_RESTRUCTURED.ipynb`.
Đã chạy đủ 78 code cell trong kernel mới, 61.5 phút, không có output lỗi. Giữ 22 chương, 101 cell và các định nghĩa lõi OMP/CNN; ba hình Mục 20.1 vẫn ở ba cell riêng. Linear là đối chứng phụ.

## Bộ trọng số mới và kiểm toán trước test

45/45 checkpoint có contract/fingerprint, history đầy đủ, epoch tốt nhất và lý do kết thúc hợp lệ. Cùng SHA của `filtered_bvp_v2.npz`, 5-fold theo đề cương, thống kê Float64 từ mẫu train duy nhất, áp dụng Float32, Phi và lượng tử upward/half-away, batch64/Adam/lr0.001/max100/patience10. Hash mọi file trọng số giữ nguyên từ trước Run All tới sau kiểm chứng. Không nhập cache cũ, không dùng adapter chuẩn hóa, không tự huấn luyện lại Lite.

Best-state validation được tái hiện cho cả 45 model trên dữ liệu canonical trước khi suy luận test. Sai khác lớn nhất: 1.43833963e-06 điểm phần trăm, trong ngưỡng 0,01 đã công bố. Hash phép chiếu Float32 của Colab và máy hiện tại khác nhau, đều được giữ nguyên và xuất audit. Colab lưu chỉ số tính Float32 còn replay tổng hợp Float64; không quy tất cả sai khác chỉ số cho BLAS, không gọi là bit-exact hoặc tự đổi fingerprint để khớp.

Các hash model/trainer trong contract Colab là hằng số tham chiếu trong nguồn đã cung cấp. Notebook Colab được lưu SHA độc lập; không coi metadata này là chứng thực độc lập toàn bộ quá trình train hay hai trainer có source giống từng byte. History và validation hỗ trợ trực tiếp việc đối chiếu trọng số. GPU train được ghi từ checkpoint: NVIDIA RTX PRO 6000 Blackwell Server Edition; tổng thời gian ghi nhận 2016.75 giây cho 45 lượt. Không dùng tổng này để so tốc độ với GPU/máy khác; peak GPU là trường được checkpoint công bố, không phải phép đo lại tại đây.

## Kết quả tính từ prediction mới

| M | OMP–DCT PRD (%) | CNN 1D PRD (%) | ResLinCNN_Lite PRD (%) |
|---:|---:|---:|---:|
| 51 | 69.41 | 22.02 | 10.06 |
| 77 | 43.01 | 19.73 | 2.33 |
| 128 | 4.25 | 19.78 | 1.19 |

Mỗi M của CNN/Lite đủ 15 lượt, 15 người. Trung bình ba seed trong từng người rồi lấy macro của 15 người; Std giữa người và biến thiên seed báo riêng. Đã tính độc lập lại PRD từ 45 file prediction Lite và cửa sổ tham chiếu canonical để đối chiếu bảng. Các trường hợp CNN kém OMP được giữ nguyên. Hình dạng sóng dùng cùng cửa sổ p25/p50/p75/p99 theo CNN gốc, không chọn lại để ưu tiên model mới.

Model được thêm sau khi đã xem test gốc, nên đây vẫn là mở rộng kiến trúc thăm dò. Cùng protocol dữ liệu và ngân sách không tự chứng minh nguyên nhân cải thiện; cần ablation khóa trước hoặc holdout mới cho kết luận kiến trúc mạnh hơn.

## CPU và ESP32 thật

Đã đo 27.000 mẫu decoder CPU: batch1, một luồng, 1.000 đầu vào/phương pháp/M/phiên, ba phiên luân phiên thứ tự. Phiên ESP32 `20261005T104005Z_7611453d`: 3.000 frame benchmark, 96 warmup và 120 trace; 3.216 packet lưu đĩa đều được kiểm lại CRC và giải lượng tử. Đã đo RAM với firmware baseline cùng bo: RAM reservation tăng thêm 3976 byte trong phạm vi phiên này. Encoder firmware được phục hồi và xác minh INFO thật tại COM5.

Timer host từ trước gửi chunk BVP thô tới packet/Lite output có UART, log và instrumentation; không bao gồm thời gian chờ lấy mẫu. Cold start khoảng 9 giây và hop 2 giây được suy ra từ Fs/drop/N/H, không phải phép đo cảm biến live. Byte UART có file capture/SHA; byte ứng dụng hữu hạn tính từ độ dài dữ liệu thật. Không suy ra radio, năng lượng, HRV hoặc độ chính xác lâm sàng.

## Bằng chứng và chạy lại

- 182 kiểm thử đạt; log `results/canonical_lite_pytest.log`.
- `canonical_lite_final_verification.json`: layout, đủ code cell, checkpoint SHA, prediction, PRD tính độc lập, CPU, packet và phiên phần cứng.
- `exports/20261005_canonical_lite_verified/canonical_lite_evaluation_manifest.json`: cache, nguồn Colab, fingerprint, history/validation, module SHA và phạm vi số học.
- Kết quả, hình và export cùng nằm dưới run ID `20261005_canonical_lite_verified`; các run trước là lịch sử với bộ trọng số khác.
- Chạy lại: `python -X utf8 execute_canonical_lite_notebook.py`; kiểm chứng: `python -X utf8 verify_canonical_lite_artifacts.py` và `python -X utf8 verify_canonical_lite_firmware_restore.py`.
- Bản trước sửa ở `backups/20261005_before_canonical_lite_pipeline/`; hướng dẫn `README_RESTRUCTURED.md`.

Giới hạn còn lại: kiến trúc bổ sung sau khi đã xem test, chưa có ablation cô lập kiến trúc/holdout mới, chưa kiểm chứng radio/năng lượng/HRV hoặc tái lập trên máy sạch. Kiểm toán quá độ là hậu nghiệm trên train; condition number là kiểm toán mẫu đã công bố. Không còn giới hạn thiếu fingerprint/history hoặc dùng cache cũ đối với bộ 45 checkpoint hiện tại.
