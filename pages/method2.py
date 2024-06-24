import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import time

    
def process_frame(frame, model_rider, model_helmet, threshold_1, threshold_2):
    final_res = [0, 0]

    frame_np = np.array(frame)

    with torch.no_grad():
        result_rider = model_rider(frame_np, conf = threshold_1)

    final_res[0] = len(result_rider[0].boxes)

    if (final_res[0] == 0):
        return frame_np, final_res
    else:
        boxes = result_rider[0].boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(frame_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cropped_image_np = frame_np[int(y1):int(y2), int(x1):int(x2)]

            # cropped_image = frame.crop((int(x1), int(y1), int(x2), int(y2)))
            # cropped_image_np = np.array(cropped_image)
            if selected_option_2 == "Image":
                st.image(cropped_image_np, caption="Cropped Rider Image", use_column_width=True)

            with torch.no_grad():
                result_helmet = model_helmet(cropped_image_np, conf = threshold_2)

            if(len(result_helmet[0].boxes) != 0):
                #print(len(result_helmet[0].boxes))
                for i, boxes in enumerate(result_helmet[0].boxes.xyxy):
                    x1_h, y1_h, x2_h, y2_h = boxes.tolist()
                    if result_helmet[0].boxes[i].cls[0] == 0:
                        cv2.rectangle(frame_np, (int(x1 + x1_h), int(y1 + y1_h)), (int(x1 + x2_h), int(y1 + y2_h)), (0, 255, 0), 2)
                    elif result_helmet[0].boxes[i].cls[0] == 1:
                        cv2.rectangle(frame_np, (int(x1 + x1_h), int(y1 + y1_h)), (int(x1 + x2_h), int(y1 + y2_h)), (255, 0, 0), 2)
                        final_res[1] += 1
                    #print(result_helmet[0].boxes[i].cls)
        return frame_np, final_res

# Tải các mô hình
# model = YOLO('best.pt')
rider = YOLO('rider.pt')
helmet = YOLO('helmet2.pt')

st.title("Helmet Detection Application")

# options_1 = ["Method_1", "Method_2"]
options_2 = ["Image", "Video"]

# selected_option_1 = st.selectbox("Select method", options_1)
selected_option_2 = st.selectbox("Select input type", options_2)

threshold_1 = st.slider("Confidence threshold rider", 0.0, 1.0, 0.5, 0.1)

threshold_2 = st.slider("Confidence threshold helmet", 0.0, 1.0, 0.5, 0.1)

if selected_option_2 == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        img_res, final_res = process_frame(image, rider, helmet, threshold_1, threshold_2)

        st.image(img_res, caption="Processed Image with Bounding Boxes", use_column_width=True)
        st.warning(f"Number of people without helmet: {final_res[1]}")
elif selected_option_2 == "Video":
    video = st.file_uploader("Choose a video...", type=["mp4"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        path = tfile.name
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FPS, 1)
        stframe = st.empty()
        placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading frame")
                break
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_res, final_res = process_frame(frame_rgb, rider, helmet, threshold_1, threshold_2)

                stframe.image(img_res, caption="Processed Image with Bounding Boxes", use_column_width=True)
                placeholder.text(f"Number of people without helmet: {final_res[1]}")
        cap.release()
        cv2.destroyWindow