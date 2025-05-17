
# 🍱 Dự án Nhận Diện Món Ăn & Tính Tiền Tự Động

## 🧾 Tổng Quan Dự Án

Dự án này nhằm xây dựng một hệ thống nhận diện các món ăn trong khay cơm và tự động tính tổng tiền dựa trên mô hình YOLO để phát hiện món ăn và CNN để phân loại. Hệ thống có thể hoạt động với ảnh tải lên hoặc kết nối trực tiếp với camera. Kết quả cuối cùng hiển thị danh sách món ăn, giá từng món và tổng hóa đơn.

---

## ⚙️ Hướng Dẫn Cài Đặt

1. **Tạo môi trường ảo (tuỳ chọn):**
   ```bash
   python -m venv yoloenv
   yoloenv\Scripts\activate  # Windows
   source yoloenv/bin/activate  # Linux/Mac
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

   Nếu không có file `requirements.txt`, dùng lệnh sau:
   ```bash
   pip install ultralytics tensorflow opencv-python pillow numpy
   ```

3. **Tải model:**
   - YOLOv8: `yolov8n.pt` hoặc `model.pt` tuỳ phiên bản bạn chọn.
   - CNN: `cnn_food.tflite` hoặc `.keras` đã huấn luyện.

---

## ▶️ Hướng Dẫn Sử Dụng

### Cách chạy chương trình bằng file ảnh:
```bash
python main.py --image path/to/image.jpg
```

### Cách chạy giao diện đồ họa (GUI):
```bash
python gui.py
```

### Đầu vào:
- Ảnh món ăn hoặc ảnh khay cơm chụp thực tế.

### Đầu ra:
- Mỗi ảnh đầu vào sẽ được phân tích thành các món ăn.
- In ra danh sách món ăn, giá từng món, và tổng cộng số tiền.

---

## 📦 Các Phần Phụ Thuộc

| Thư Viện         | Chức Năng                        |
|------------------|----------------------------------|
| `ultralytics`    | Dùng để chạy mô hình YOLOv8      |
| `tensorflow`     | Dùng để chạy mô hình CNN         |
| `opencv-python`  | Xử lý ảnh và kết nối camera      |
| `Pillow`         | Đọc ảnh và chuyển đổi định dạng  |
| `numpy`          | Xử lý mảng và toán học ma trận   |
| `tkinter` hoặc `gradio` | Xây dựng giao diện (tùy chọn) |

---

## ✅ Chất Lượng Chương Trình

- ✅ **Cấu trúc rõ ràng**: Code chia thành các module riêng biệt như `detect.py`, `classify.py`, `gui.py`, `utils.py`.
- ✅ **Dễ bảo trì**: Biến, hàm được đặt tên rõ nghĩa, dễ hiểu.
- ✅ **Tuân thủ PEP8**: Mã nguồn được định dạng sạch, dễ đọc.
- ✅ **Có chú thích**: Giải thích rõ từng bước xử lý và logic.

---

## 📬 Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý, xin vui lòng liên hệ qua GitHub hoặc email hỗ trợ dự án.
