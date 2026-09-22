# Nhận diện biển báo giao thông bằng YOLO

Ứng dụng desktop nhận diện **43 loại biển báo giao thông** từ ảnh, webcam hoặc video. Dự án sử dụng Ultralytics YOLO để huấn luyện và suy luận, OpenCV để xử lý hình ảnh, cùng Tkinter/ttkbootstrap để xây dựng giao diện.

## Chức năng chính

- Nhận diện biển báo trong ảnh (`.jpg`, `.jpeg`, `.png`).
- Nhận diện thời gian thực qua webcam.
- Nhận diện biển báo trong video (`.mp4`, `.avi`, `.mov`, `.mkv`).
- Hiển thị tên lớp, độ tin cậy và bounding box của từng biển báo.
- So sánh kết quả của hai checkpoint trên cùng một đầu vào.
- Huấn luyện YOLO theo hai giai đoạn và tiếp tục từ checkpoint gần nhất.
- Trực quan hóa nhãn YOLO và tạo một bản dataset giới hạn số mẫu theo lớp.

## Công nghệ sử dụng

- Python 3.9
- Ultralytics YOLO
- PyTorch
- OpenCV
- Pillow
- Tkinter và ttkbootstrap
- tqdm

## Cấu trúc dự án

```text
xulianh/
├── app.py                  # Ứng dụng nhận diện bằng một model
├── app5.py                 # Ứng dụng so sánh hai model
├── data.yaml               # Cấu hình dataset dùng để huấn luyện
├── data2.yaml              # Cấu hình dataset giai đoạn/bộ dữ liệu thứ hai
├── train_stage1.py         # Huấn luyện giai đoạn 1 từ yolov8s.pt
├── train_stage2.py         # Fine-tune từ best.pt của giai đoạn 1
├── train_resume.py         # Tiếp tục giai đoạn 1 từ last.pt
├── train_resume2.py        # Tiếp tục giai đoạn 2 từ last.pt
├── visualize_labels.py     # Vẽ bounding box từ nhãn YOLO
├── del_dataset.py          # Tạo dataset_balanced từ dataset hiện có
├── dataset/                # Dataset chính (train/val/test)
├── dataset_S2/             # Dataset thứ hai
├── Meta/                   # Ảnh minh họa cho 43 lớp
└── runs/detect/            # Checkpoint và kết quả huấn luyện
```

Thư mục `source/` chứa bản sao của một số script cũ. Các lệnh trong tài liệu này sử dụng những file ở thư mục gốc.

## Yêu cầu

- Windows 10/11.
- Python 3.9 trở lên (môi trường hiện tại của dự án dùng Python 3.9.13).
- Tkinter, thường được cài kèm Python trên Windows.
- GPU NVIDIA và CUDA là tùy chọn khi chạy ứng dụng, nhưng các script huấn luyện hiện đặt `device=0` nên cần GPU CUDA hoặc phải đổi cấu hình thiết bị.

## Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python pillow ttkbootstrap tqdm
```

Nếu PowerShell chặn script kích hoạt môi trường, có thể chạy Python trực tiếp bằng đường dẫn `.\.venv\Scripts\python.exe`.

## Chạy ứng dụng

### Nhận diện bằng một model

`app.py` mặc định đọc checkpoint:

```text
runs/detect/YOLO_Stage2/weights/best.pt
```

Khởi chạy:

```powershell
python app.py
```

Ứng dụng gồm ba tab:

- **Ảnh:** chọn một ảnh từ máy tính.
- **Webcam:** nhấn **Bật camera** để nhận diện trực tiếp.
- **Video:** chọn video có sẵn và theo dõi kết quả theo từng khung hình.

### So sánh hai model

`app5.py` chạy đồng thời hai checkpoint:

```text
runs/detect/YOLO_Stage2/weights/best.pt
runs/detect/yolov5_run/weights/best.pt
```

Khởi chạy:

```powershell
python app5.py
```

Hai kết quả được hiển thị cạnh nhau cho ảnh, webcam và video. Cách này cần nhiều CPU/GPU và bộ nhớ hơn `app.py`.

## Chuẩn bị dataset

Dataset sử dụng định dạng phát hiện đối tượng của YOLO:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Mỗi ảnh phải có một file nhãn `.txt` cùng tên. Mỗi dòng nhãn có dạng:

```text
class_id x_center y_center width height
```

Các tọa độ và kích thước phải được chuẩn hóa về khoảng `0` đến `1`. Danh sách đầy đủ 43 lớp nằm trong `names` của `data.yaml` và `data2.yaml`.

> **Quan trọng:** trường `path` trong hai file YAML hiện là đường dẫn tuyệt đối của máy đã tạo dataset. Hãy đổi nó thành đường dẫn dataset trên máy của bạn trước khi huấn luyện, ví dụ:

```yaml
path: C:/Users/YourName/Documents/xulianh/dataset
train: train/images
val: val/images
test: test/images
```

## Huấn luyện model

Các lệnh dưới đây cần được chạy tại thư mục gốc của dự án.

### Giai đoạn 1

Huấn luyện từ checkpoint COCO `yolov8s.pt` với ảnh kích thước 640:

```powershell
python train_stage1.py
```

Kết quả được lưu tại `runs/detect/YOLO_Stage1/`.

### Giai đoạn 2

Fine-tune từ `runs/detect/YOLO_Stage1/weights/best.pt` với ảnh kích thước 768:

```powershell
python train_stage2.py
```

Kết quả được lưu tại `runs/detect/YOLO_Stage2/`.

### Tiếp tục một phiên huấn luyện

```powershell
python train_resume.py
python train_resume2.py
```

Hai script lần lượt đọc `last.pt` của giai đoạn 1 và giai đoạn 2. Chỉ chạy khi checkpoint tương ứng đã tồn tại.

### Huấn luyện không dùng GPU CUDA

Các script hiện đặt:

```python
device=0
```

Để huấn luyện bằng CPU, đổi thành `device="cpu"`. Quá trình sẽ chậm hơn đáng kể.

## Công cụ hỗ trợ

### Kiểm tra nhãn trực quan

`visualize_labels.py` đọc `dataset_S2`, vẽ bounding box và ghi ảnh kết quả vào `dataset_S2_visualized`:

```powershell
python visualize_labels.py
```

Có thể thay `DATASET_PATH` và `OUTPUT_PATH` ở đầu file nếu muốn dùng thư mục khác.

### Tạo dataset giới hạn số mẫu theo lớp

`del_dataset.py` gom dữ liệu train/val, giữ tối đa 600 mẫu cho mỗi lớp, chia lại train/val theo tỷ lệ 80/20 và sao chép test sang `dataset_balanced`:

```powershell
python del_dataset.py
```

Script chỉ sao chép dữ liệu, không xóa dataset gốc. Hãy kiểm tra các hằng số `SRC_DATASET`, `NEW_DATASET`, `MAX_PER_CLASS` và `VAL_RATIO` trước khi chạy.

## Tùy chỉnh

Các thiết lập suy luận chính nằm ở đầu `app.py` và `app5.py`:

```python
MODEL_PATH = r"runs/detect/YOLO_Stage2/weights/best.pt"
FONT_PATH = "Arial.Unicode.ttf"
CONF_THRES = 0.2
```

- Đổi đường dẫn model nếu checkpoint được đặt ở vị trí khác.
- Tăng `CONF_THRES` để giảm các dự đoán có độ tin cậy thấp.
- Nếu font chú thích không tải được, đặt `FONT_PATH` thành đường dẫn tới một font `.ttf` hỗ trợ tiếng Việt trên máy.

## Lỗi thường gặp

### Không thể tải model

Kiểm tra file `best.pt` có tồn tại đúng đường dẫn được khai báo trong ứng dụng. Nếu chưa có checkpoint, cần huấn luyện model hoặc chép checkpoint vào đúng vị trí.

### Không mở được webcam

Đóng các ứng dụng khác đang sử dụng camera, cấp quyền camera cho Python trong Windows Settings rồi chạy lại ứng dụng.

### CUDA không khả dụng

Kiểm tra bản PyTorch, driver NVIDIA và CUDA tương thích. Nếu chỉ muốn thử chương trình, dùng CPU hoặc đổi `device` trong script huấn luyện.

### Không tìm thấy ảnh hoặc nhãn khi train

Kiểm tra lại `path` trong `data.yaml`, cấu trúc các thư mục `images`/`labels`, và bảo đảm tên file ảnh trùng với tên file nhãn.

## Lưu ý

Dự án phục vụ học tập và thử nghiệm nhận diện hình ảnh. Kết quả dự đoán không nên được dùng làm nguồn duy nhất cho các quyết định an toàn giao thông.
