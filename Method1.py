import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import time

# def draw_bounding_boxes(image, predictions):
#     draw = ImageDraw.Draw(image)
#     for prediction in predictions:
#         x = prediction['x']
#         y = prediction['y']
#         width = prediction['width']
#         height = prediction['height']
#         confidence = prediction['confidence']
#         class_name = prediction['class']

#         # Calculate coordinates
#         left = x - width / 2
#         top = y - height / 2
#         right = x + width / 2
#         bottom = y + height / 2

#         # Draw bounding box
#         color = "blue"
#         if class_name == 'rider':
#             color = "yellow"
#         elif class_name == 'helmet':
#             color = "green"
#         elif class_name == 'no-helmet':
#             color = "red"

#         draw.rectangle([left, top, right, bottom], outline=color, width=3)
#         # Draw label
#         draw.text((left, top), f"{class_name} {confidence:.2f}", fill="blue")

#     return image

def draw_bounding_boxes(image, results):
    count = 0
    draw = ImageDraw.Draw(image)
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        label = names[cls]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
        if label == 'no-helmet':
            count += 1
    st.warning(f"Số người không đội mũ bảo hiểm: {count}")
    return image

# Tải các mô hình
model = YOLO('best.pt')




st.title("Helmet Detection Application")

# Tải video từ người dùng
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"], accept_multiple_files=False)

if uploaded_image is not None:
    if uploaded_image is not None:
        time_start = time.time()
        image = Image.open(uploaded_image).convert('RGB')
        image_np = np.array(image)
    
        # Chạy mô hình để phát hiện
        results = model(image_np)
        
        # Vẽ bounding boxes lên ảnh
        image_with_boxes = draw_bounding_boxes(image, results[0])
        time_end = time.time()
        
        # Hiển thị ảnh với bounding boxes
        st.image(image_with_boxes, caption="Processed Image with Bounding Boxes", use_column_width=True)
        st.warning(f"Thời gian xử lý: {time_end - time_start} giây")

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Đọc video
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển đổi khung hình từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Chạy mô hình để phát hiện
        results = model(frame_rgb)
        
        # Vẽ bounding boxes lên ảnh
        frame_with_boxes = draw_bounding_boxes(frame_pil, results[0])
        
        # Hiển thị khung hình đã xử lý
        stframe.image(frame_with_boxes, caption="Processed Frame with Bounding Boxes", use_column_width=True)

    cap.release()
