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

def draw_bounding_boxes2(image, results):
    count = 0
    draw = ImageDraw.Draw(image)
    names = results.names

    for box in results.boxes:
        bounding_boxes = []
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        label = names[cls]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
        if label == 'no-helmet':
            bounding_boxes.append((x1, y1, x2, y2))
            count += 1
                
    st.warning(f"Số người không đội mũ bảo hiểm: {count}")
    return image

def draw_bounding_boxes(image, results):
    draw = ImageDraw.Draw(image)
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        label = names[cls]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
    return image
# Hàm để cắt bounding box từ ảnh
def crop_bounding_boxes(image, results, class_name):
    names = results.names
    cropped_images = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        if names[cls] == class_name:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)
    
    return cropped_images
def check_helmet_position(helmet_box, image_height):
    # Tính toán tọa độ y của bounding box của mũ bảo hiểm
    helmet_y = (helmet_box[1] + helmet_box[3]) / 2
    
    # So sánh tọa độ y với giữa 0 và chiều cao của ảnh
    if helmet_y - (image_height*2/3)>0:
        return True  # Nếu bounding box nằm ở nửa trên của ảnh
    else:
        return False  # Nếu bounding box nằm ở nửa dưới của ảnh

# Tải các mô hình
model = YOLO('best.pt')
rider = YOLO('rider.pt')
helmet = YOLO('helmet2.pt')




st.title("Helmet Detection Application")

# Tải video từ người dùng
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_image is not None:
    if uploaded_image is not None:
        time_start = time.time()
        image = Image.open(uploaded_image).convert('RGB')
        image_np = np.array(image)
    
        # Chạy mô hình để phát hiện
        results = rider(image_np)
        
        # kiểm tra có rider hay không
        if len(results) == 0:
            st.warning("Không có người trong ảnh")
        else:
            # Vẽ bounding boxes lên ảnh
            image_with_boxes = draw_bounding_boxes(image, results[0])
            
            # Hiển thị ảnh với bounding boxes
            st.image(image_with_boxes, caption="Processed Image with Bounding Boxes", use_column_width=True)

            # detect helmet
            # Cắt bounding boxes cho class 'rider'
            cropped_rider_images = crop_bounding_boxes(image, results[0], 'rider')
            # Hiển thị ảnh đã cắt
            st.subheader("Cropped Rider Images")
            for i, cropped_image in enumerate(cropped_rider_images):
                st.image(cropped_image, caption=f"Cropped Rider Image {i+1}", use_column_width=True)
            # st.write("Cropped Rider Images111")
            # st.image(cropped_image)

            # st.write(cropped_image.size)
            crop_img_np = np.array(cropped_image)
            results2 = helmet(crop_img_np)
            # st.write(results2)

            if len(results2) == 0:
                st.warning("Không có người đội mũ bảo hiểm")
            else:
                # helmet_box = results2.boxes[0]
                # if check_helmet_position(helmet_box, cropped_image.height):
                    image_with_boxes2 = draw_bounding_boxes2(cropped_image, results2[0])
                    time_end = time.time()
                    st.image(image_with_boxes2, caption="Processed Image with Bounding Boxes", use_column_width=True)
                    st.warning(f"Thời gian xử lý: {time_end - time_start:.2f} giây")

        
