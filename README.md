# Đề tài C9 — Nén cảm nhận tín hiệu PPG

**Sinh viên:** Phùng Lê Thanh Quân — **MSSV:** 23110145  
**Học phần:** Trí tuệ nhân tạo cho IoT  
**Dữ liệu thực nghiệm:** PPG-DaLiA, 15 đối tượng, tín hiệu BVP 64 Hz  
**Thiết bị:** ESP32-S3

Đề tài so sánh OMP–DCT và CNN 1D để tái tạo tín hiệu PPG sau nén cảm nhận. Notebook có thêm khảo sát ResLinCNN_Lite và đối chứng Linear. Thiết bị ESP32 thực hiện tiền xử lý, nén và đóng gói; các bộ giải mã chạy trên máy tính.

## Nội dung bộ nộp

| Thành phần | File hoặc thư mục |
|---|---|
| Báo cáo PDF & Word | `PhungLeThanhQuan_C9_BaoCao.pdf`, `PhungLeThanhQuan_C9_BaoCao.docx` |
| Slide thuyết trình | `PhungLeThanhQuan_C9_Slide_ThuyetTrinh.pptx` |
| Notebook có output thực nghiệm | `PhungLeThanhQuan_C9_Notebook.ipynb` |
| Notebook Colab huấn luyện phụ | `ResLinCNN_Lite_Model_Mở Rộng (train riêng để tối ưu thời gian).ipynb` |
| Module Python cốt lõi tái lập Notebook | `src/` (19 module thuật toán CS, OMP-DCT, CNN 1D, ESP32, kiểm toán) |
| Các gói Python | `requirements.txt` |
| Mã firmware và cấu hình PlatformIO | `esp32_firmware/src/`, `esp32_firmware/platformio.ini` |
| Bảng số liệu, lịch sử huấn luyện và log đo | `results/20261005_canonical_lite_verified/` |

Bộ nộp đã bao gồm đầy đủ **19 module Python trong `src/`** để Notebook có thể import và chạy tái lập các thuật toán. Các trọng số huấn luyện (.pt) và tập dữ liệu lớn được lưu trữ trên Google Drive do giới hạn kích thước nộp bài.

Để giảm dung lượng khi nộp LMS, 138 file `.npz` chứa dự đoán và mảng tín hiệu trong `results/` đã được loại khỏi bản này. Các bảng CSV, history, JSON, log và output nhúng trong notebook được giữ lại. Các đường dẫn tới `.npz` trong metadata là tham chiếu của lần chạy gốc; bản gọn không cung cấp các mảng đó để kiểm tra lại từng cửa sổ. File `.npz.json` chỉ là metadata, không thay thế file `.npz` đã loại.

## Cách xem kết quả

1. Đọc báo cáo Word.
2. Mở notebook bằng trình xem Jupyter Notebook hoặc VS Code có hỗ trợ notebook. Các bảng và hình đã được lưu trong output, không cần bấm **Run All** để xem kết quả.
3. Đối chiếu các file trong `results/20261005_canonical_lite_verified/`:

| File | Nội dung |
|---|---|
| `table8_main_summary.csv` | So sánh OMP–DCT, CNN 1D và ResLinCNN_Lite tại M = 51, 77, 128 |
| `table_lite_completion_quality.csv` | Mức hoàn thành và chất lượng tái tạo của Lite |
| `table_lite_per_subject.csv` | Kết quả Lite theo đối tượng |
| `lite_full_training_history.csv` | Lịch sử theo epoch của 45 lượt huấn luyện Lite |
| `lite_training_completion.csv` | Epoch tốt nhất, epoch đã chạy và thông tin kết thúc huấn luyện |
| `canonical_lite_validation_audit.csv` | Đối chiếu validation của các checkpoint Lite |
| `hardware_timings.csv`, `ram_comparison.json` | Thời gian xử lý và bộ nhớ đo trên ESP32 |
| `environment.json` | Môi trường của lần chạy thực nghiệm |

Báo cáo trình bày so sánh OMP–DCT/CNN theo đề cương; kết quả định lượng của phần mở rộng Lite xem trong notebook và các CSV tương ứng. Các đường dẫn tuyệt đối và cổng COM trong log/JSON ghi nhận máy thực nghiệm ban đầu.

## Môi trường Python

Phiên thực nghiệm đã lưu dùng **Python 3.12.10**, NumPy 2.2.6, SciPy 1.14.1 và PyTorch 2.6.0+cu124. `requirements.txt` giữ phiên bản các gói đang có trên máy thực nghiệm; riêng PyTorch ghi `2.6.0` để có thể chọn bản CPU hoặc CUDA phù hợp.

Ví dụ tạo môi trường trên Windows, chạy terminal tại thư mục bộ nộp:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name ppg-c9 --display-name "Python PPG C9"
```

Chọn kernel **Python PPG C9** khi mở notebook. Nếu cần môi trường CUDA 12.4 như phiên thực nghiệm, thay lệnh cài PyTorch CPU bằng:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Không cần cài Python chỉ để đọc báo cáo hoặc xem output bằng trình xem notebook có sẵn. Thời gian benchmark đã lưu thuộc máy đo ban đầu, không đại diện cho thời gian trên mọi máy.

## Điều kiện để chạy lại thực nghiệm

Cần bản dự án đầy đủ, giữ nguyên cấu trúc gồm:

- Các module Python hỗ trợ, ví dụ `experiment_config.py`, `provenance.py`, `dataset.py`, `train.py`, `notebook_lite_extension.py`, `canonical_lite_audits.py`, `notebook_scientific_audits.py`, `eda_hardware.py` và các module mà chúng sử dụng.
- Dữ liệu PPG-DaLiA gốc `S1/S1.pkl` đến `S15/S15.pkl`; cache `data/filtered_bvp_v2.npz` và metadata tương ứng.
- 45 checkpoint Lite trong `checkpoints/`, 45 checkpoint CNN và 5 checkpoint Linear trong `checkpoints/20261004_three_models_v2/`, cùng metadata/lịch sử đi kèm.
- ESP32-S3 và PlatformIO cho các cell đo phần cứng. Firmware đo baseline RAM cũng cần được cung cấp khi tái lập phép đo so sánh RAM.

Trước khi mở Jupyter/VS Code từ terminal, cấu hình các đường dẫn thực tế, ví dụ:

```powershell
$env:PPG_PROJECT_ROOT = "D:\DuAnPPGDayDu"
$env:PPG_DATA_ROOT = "D:\Dataset\PPG_FieldStudy"
$env:PPG_LITE_CHECKPOINT_DIR = "D:\DuAnPPGDayDu\checkpoints"
```

Sửa cổng COM trong cell tạo `ExperimentConfig` của notebook và trong `esp32_firmware/platformio.ini` theo máy chạy. Notebook hiện đặt COM5, còn file PlatformIO đặt COM6. Sao lưu kết quả trước khi chạy lại vì các cell có ghi file kết quả. Với riêng bộ nộp gọn này, **Run All sẽ dừng do thiếu module hỗ trợ**.

## Firmware

Firmware được cấu hình bằng PlatformIO, platform `espressif32@6.13.0`, board `esp32-s3-devkitc-1`, framework **Arduino**. PlatformIO và toolchain được cài riêng, không nằm trong `requirements.txt`.

Khi đã có PlatformIO và bo mạch phù hợp:

```powershell
pio run -d esp32_firmware
pio run -d esp32_firmware -t upload --upload-port COM6
pio device monitor --port COM6 --baud 115200
```

Thay `COM6` bằng cổng thực tế. Chỉ upload khi đã kết nối đúng bo mạch.

## Phạm vi kết luận

Đánh giá phần mở rộng Lite mang tính thăm dò vì kết quả test của pipeline gốc đã được xem trước đó. Các chỉ số phản ánh chất lượng tái tạo tín hiệu và phép đo xử lý/truyền dữ liệu; không suy diễn thành kết quả chẩn đoán y tế hoặc mức tiết kiệm điện năng khi chưa có phép đo tương ứng.
