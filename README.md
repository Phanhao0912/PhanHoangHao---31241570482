
# 🥗 HỆ THỐNG NHẬN DIỆN MÓN ĂN VÀ TÍNH TIỀN TỰ ĐỘNG

Một hệ thống AI giúp tự động phát hiện các món ăn từ khay cơm, phân loại và tính tổng hóa đơn. Sử dụng YOLOv8 để phát hiện vật thể (tô/món ăn) và CNN để phân loại từng món cụ thể. Giao diện người dùng được thiết kế bằng Gradio trực quan, chạy trên Kaggle Notebook.

---

## 🚀 MỤC TIÊU

- Tự động nhận diện các món ăn trong ảnh chụp khay cơm.
- Phân loại món ăn bằng mô hình CNN đã huấn luyện.
- Tính tiền từ file `menu.json` dựa theo từng món ăn đã phân loại.
- Hiển thị kết quả trực quan với Gradio GUI.

---

## ⚙️ HƯỚNG DẪN CÀI ĐẶT (TRÊN KAGGLE)

### 1. Bật GPU trong phần "Accelerator"
- Chọn **Notebook Settings > Accelerator: GPU**

### 2. Upload các file cần thiết:
- `best_food_model.keras` – mô hình CNN phân loại món ăn.
- `model.pt` – mô hình YOLO phát hiện món ăn trong từng tô.
- `yolo11n.pt` – mô hình YOLO phát hiện các tô trên khay.
- `menu.json` – file JSON chứa tên món ăn và giá tiền.
- `*.jpg` – ảnh khay cơm để nhận diện.

### 3. Cài đặt thư viện trong ô đầu tiên:
```python
!pip install -q ultralytics gradio tensorflow pillow opencv-python matplotlib
```

---

## ▶️ HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH

### 📍 Các bước xử lý:
1. **YOLO 1 (`yolo11n.pt`)**: Phát hiện các tô trên khay.
2. **YOLO 2 (`model.pt`)**: Phát hiện các món ăn trong từng tô.
3. **CNN (`best_food_model.keras`)**: Phân loại món ăn đã crop.
4. **Gradio**: Hiển thị ảnh, tên món, giá từng món và tổng tiền.

### 📌 Hàm chính được gọi bởi nút Gradio:
```python
def process_all_dishes(_):
    ...
```

### 🖼️ Đầu ra:
- Thư mục `dishes_cropped/` chứa các món đã tách.
- Giao diện Gradio gồm:
  - Thư viện ảnh các món.
  - Danh sách món + giá.
  - Hóa đơn tổng kết.

---

## 📂 CẤU TRÚC THƯ MỤC

```
/kaggle/input/
├── allmodeltrain/
│   ├── best_food_model.keras
│   ├── model.pt
│   └── yolo11n.pt
├── menuueh/
│   └── menu.json
├── foodtestproject/
│   └── khay_com.jpg
```

---

## 🧾 MẪU FILE `menu.json`

```json
{
  "cahukho": 12000,
  "canhcai": 8000,
  "canhchua": 9000,
  "com": 5000,
  "dauhusotca": 10000,
  "gachien": 15000,
  "raumuongxao": 7000,
  "thitkho": 11000,
  "thitkhotrung": 13000,
  "trungchien": 6000
}
```

---

## ✅ YÊU CẦU

| Thư viện         | Phiên bản khuyến nghị |
|------------------|------------------------|
| ultralytics      | >=8.0.100              |
| tensorflow       | >=2.10.0               |
| opencv-python    | >=4.7.0                |
| pillow           | >=9.0.0                |
| gradio           | >=4.0.0                |
| numpy            | >=1.23                 |

---

## 📬 LIÊN HỆ

Mọi thắc mắc vui lòng liên hệ qua GitHub hoặc bình luận trực tiếp trên Kaggle Notebook.

---

**Chúc bạn triển khai thành công!** 🎉
