
# ===========================================
# 🍱 HỆ THỐNG NHẬN DIỆN MÓN ĂN + TÍNH TIỀN
# ===========================================

# CÀI ĐẶT THƯ VIỆN CẦN THIẾT
!pip install -q ultralytics gradio tensorflow pillow opencv-python matplotlib

# ==== PHẦN 1: YOLO - Phát hiện món ăn ====

import os
import cv2
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# Thiết lập môi trường lưu log YOLO
os.environ['YOLO_CACHE_DIR'] = '/kaggle/working/yolo_cache'
SETTINGS.update({'runs_dir': '/kaggle/working/ultralytics_logs'})

# Đường dẫn ảnh và thư mục crop
image_path = "/kaggle/input/foodtestproject/z6582828766338_58b7500b77e640b493268491c92456e0.jpg"
image_name = os.path.splitext(os.path.basename(image_path))[0]
bowl_crop_dir = "/kaggle/working/bowls_cropped"
dish_crop_dir = "/kaggle/working/dishes_cropped"

# Xóa dữ liệu cũ
for path in [bowl_crop_dir, dish_crop_dir]:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

# Load mô hình YOLO
bowl_model = YOLO(shutil.copy("/kaggle/input/allmodeltrain/yolo11n.pt", "/kaggle/working/yolo11n.pt"))
dish_model = YOLO(shutil.copy("/kaggle/input/allmodeltrain/model.pt", "/kaggle/working/model.pt"))

# Đọc ảnh khay
img = cv2.imread(image_path)

# Nhận diện các tô món ăn
bowl_results = bowl_model.predict(source=image_path, conf=0.3)

print("🔍 Đã phát hiện các tô món ăn:")
for r in bowl_results:
    for idx, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        class_name = bowl_model.names[cls_id].lower()
        conf = float(box.conf[0])
        if "bowl" not in class_name:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        crop_name = f"{image_name}_bowl_{idx}_conf{conf:.2f}.jpg"
        cv2.imwrite(os.path.join(bowl_crop_dir, crop_name), crop)

# Tách món từ mỗi tô
print("\n🍽️ Đang nhận diện món ăn từ các tô...")

for fname in sorted(os.listdir(bowl_crop_dir)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    bowl_path = os.path.join(bowl_crop_dir, fname)
    bowl_img = cv2.imread(bowl_path)
    bowl_name = os.path.splitext(fname)[0]

    dish_results = dish_model.predict(source=bowl_path, conf=0.3)

    for r in dish_results:
        for idx, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            class_name = dish_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = bowl_img[y1:y2, x1:x2]
            crop_name = f"{bowl_name}_dish_{class_name}_{idx}_conf{conf:.2f}.jpg"
            cv2.imwrite(os.path.join(dish_crop_dir, crop_name), crop)

# ==== PHẦN 2: CNN - Nhận diện & tính tiền ====

import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# Tắt sử dụng GPU
tf.config.set_visible_devices([], 'GPU')

# Load model CNN và menu
cnn_model = load_model("/kaggle/input/allmodeltrain/best_food_model.keras")
with open("/kaggle/input/menuueh/menu.json", "r", encoding="utf-8") as f:
    menu = json.load(f)

class_names = list(menu.keys())

# Hàm xử lý nhận diện toàn bộ ảnh
def process_all_dishes(_):
    results = []
    total_price = 0
    dish_images = []

    for fname in sorted(os.listdir(dish_crop_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(dish_crop_dir, fname)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = cnn_model.predict(img_array, verbose=0)
        cls_index = np.argmax(pred)
        cls_name = class_names[cls_index]
        price = menu.get(cls_name, 0)
        total_price += price

        results.append((cls_name, price))
        dish_images.append((Image.open(img_path), f"🍽️ {cls_name} – {price:,} VND"))

    receipt = "<b>===== 🧾 HÓA ĐƠN =====</b><br>"
    for name, price in results:
        receipt += f"<div style='display:flex; justify-content:space-between;'><span>{name}</span><span>{price:,} VND</span></div><br>"
    receipt += "<hr><b>TỔNG TIỀN PHẢI TRẢ: {:,} VND</b>".format(total_price)

    if dish_images:
        imgs, labels = zip(*dish_images)
    else:
        imgs, labels = [], []

    return list(imgs), "\n".join(labels), receipt

# Giao diện Gradio
custom_theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="gray")

demo = gr.Interface(
    fn=process_all_dishes,
    inputs=gr.Button("📤 NHẬN DIỆN TẤT CẢ MÓN", elem_classes="btn-primary"),
    outputs=[
        gr.Gallery(label="📷 Ảnh món ăn đã phát hiện", columns=3, height="auto"),
        gr.Textbox(label="📋 Danh sách món và giá", lines=6),
        gr.HTML(label="🧾 Hóa đơn tổng kết")
    ],
    title="🥗 HỆ THỐNG NHẬN DIỆN MÓN ĂN CĂN TIN",
    description="🚀 Dự án nhận diện các món ăn từ khay bằng YOLOv8 & phân loại bằng CNN. Tự động tính tiền dựa trên file <code>menu.json</code>.",
    theme=custom_theme,
    live=False
)

demo.launch(debug=False, share=True)
