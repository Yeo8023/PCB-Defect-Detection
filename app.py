import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="PCB Defect Detection", layout="wide")

st.title("🔍 PCB Defect Detection System")
st.write("Upload an image of a Printed Circuit Board to detect manufacturing defects.")

@st.cache_resource
def load_model():
    # Load our custom trained YOLO model
    return YOLO("runs/detect/pcb_defect_model/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(opencv_image, channels="BGR", use_container_width=True)
    
    with st.spinner("Detecting defects..."):
        # Run inference
        results = model(opencv_image)
        # Plot bounding boxes on the image
        res_plotted = results[0].plot()
    
    with col2:
        st.header("Detected Defects")
        st.image(res_plotted, channels="BGR", use_container_width=True)
    
    st.markdown("---")
    st.subheader("Detection Summary")
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.success("No defects found! 🎉 The PCB looks good.")
    else:
        st.warning(f"Found {len(boxes)} defect(s).")
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            st.write(f"- **{class_name}** (Confidence: {conf:.2%})")
