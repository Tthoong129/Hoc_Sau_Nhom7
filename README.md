# Hệ thống Giám sát Thi cử Trực tuyến (Online Proctoring System)

Đây là ứng dụng giám sát thi cử trực tuyến thời gian thực viết bằng Python. Hệ thống sử dụng Flask + Socket.IO ở frontend để truyền tải luồng hình ảnh trực tiếp, kết hợp với các mô hình thị giác máy tính YOLO và các thuật toán phân tích tư thế ở backend nhằm phát hiện hành vi vi phạm quy chế của thí sinh.

---

## Cách hoạt động & Phân cấp cảnh báo

Hệ thống phân tích hình ảnh từ Camera hoặc file Video đã tải lên để nhận diện hai nhóm hành vi:

*   **Cảnh báo Đỏ (Gian lận trực tiếp - Nhận diện bằng YOLOv8):**
    *   Phát hiện thí sinh sử dụng điện thoại, đồng hồ thông minh, tai nghe hoặc phao thi giấy.
*   **Cảnh báo Vàng (Dấu hiệu bất thường - Nhận diện qua Pose & Tracking):**
    *   Phát hiện thí sinh quay đầu nhìn xung quanh quá lâu, nhoài người ra khỏi vùng camera, hoặc liếc đi liếc lại liên tục.

### Lưu trữ bằng chứng & Tối ưu hóa hiệu năng
*   **Chụp ảnh tự động:** Khi xảy ra cảnh báo Vàng hoặc Đỏ, hệ thống sẽ tự động chụp ảnh màn hình bằng chứng và phân loại vào hai thư mục: phiên chạy hiện tại (`session`) và lịch sử (`history`).
*   **Ghi log thời gian thực:** Toàn bộ lịch sử cảnh báo được lưu dồn vào tệp `violation_log.txt` trên server để phòng ngừa sự cố mất điện hoặc sập ứng dụng đột ngột.
*   **Tua video thông minh:** Hỗ trợ xem lại video ở tốc độ cao (1.5x, 2x, 3x, 4x) thông qua cơ chế tự động bỏ qua khung hình (frame-skipping) thay vì ép phần cứng xử lý toàn bộ khung hình, giúp hệ thống vận hành êm ái trên các máy cấu hình tầm trung.

---

## Dữ liệu huấn luyện & Mô hình (Dataset & Models)

Mô hình nhận diện vật thể gian lận (YOLOv8 custom) được huấn luyện từ tập dữ liệu tổng hợp và hiệu chỉnh lại từ các nguồn mở trên Roboflow:
*   [Nguồn 1 - offline exam monitoring 4 Computer Vision Model](https://universe.roboflow.com/cp2-sgbvv/offline-exam-monitoring-4)
*   [Nguồn 2 - Exam Violation Detection Computer Vision Dataset](https://universe.roboflow.com/cgm3-8oh0v/exam-violation-detection-2sbek)
*   [Dataset tổng hợp và hiệu chỉnh hoàn thiện](https://app.roboflow.com/tan-thong/deepnew-rmes5/train)

---

## Cài đặt & Chạy ứng dụng

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.9 trở lên và một Webcam (nếu muốn giám sát trực tiếp).

### 2. Cài đặt thư viện
Mở terminal tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 3. Tệp trọng số mô hình (Weights)
Đặt các tệp mô hình vào đúng đường dẫn sau để hệ thống nhận diện:
*   Mô hình nhận diện vật thể: `models/best.pt`
*   Mô hình tư thế: `yolov8n-pose.pt`, `yolov8s-pose.pt` (nằm ở thư mục gốc dự án)

### 4. Khởi chạy ứng dụng
Chạy lệnh khởi động server:
```bash
python app.py
```
Sau đó, truy cập giao diện giám sát trên trình duyệt qua địa chỉ: `http://localhost:5000`

### 5. Chạy thử nghiệm
Các đoạn video kiểm thử (test video) đã được chuẩn bị sẵn và lưu trong thư mục `VideoTest/` của dự án. Bạn có thể sử dụng chức năng **Tải lên Video** trên giao diện web và chọn các video trong thư mục này để kiểm tra ngay khả năng bắt gian lận của hệ thống.

---

## Cấu trúc thư mục dự án

```text
├── core/                # Các module xử lý AI (Engine, Tracker, Drawing, Motion)
├── models/              # Thư mục chứa các file trọng số AI (.pt)
├── screenshots/         # Lưu hình ảnh bằng chứng vi phạm
│   ├── session/         # Ảnh chụp trong phiên giám sát hiện tại
│   └── history/         # Lưu lịch sử cảnh báo + tệp `violation_log.txt`
├── static/              # CSS, Javascript (sử dụng giao diện sáng - Light Theme)
├── templates/           # File HTML giao diện trang web
├── VideoTest/           # Thư mục chứa các video dùng để chạy thử nghiệm hệ thống
├── app.py               # File khởi chạy ứng dụng chính (Flask + Socket.IO)
└── requirements.txt     # Danh sách thư viện Python cần cài đặt
└── FileColabTrain.txt     # Danh sách thư viện Python cần cài đặt
```
### Kết quả train yolov8
<img width="886" height="460" alt="image" src="https://github.com/user-attachments/assets/230e8208-80ba-47ac-8e92-56a01d996708" />
---
### Ma trận nhầm lẫn
<img width="886" height="664" alt="image" src="https://github.com/user-attachments/assets/8db12da4-d131-41d3-aea7-9314f7ce241f" />
