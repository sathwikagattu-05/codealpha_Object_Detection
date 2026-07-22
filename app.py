import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile 
import cv2
import time

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Object Detection & Tracking")

st.markdown(
"""
Detect and track real-world objects in images and live webcam using **YOLOv8**.
"""
)
st.write("Upload an image to detect objects.")
with st.sidebar:
    confidence = st.slider(
        "Confidence",
        0.1,
        1.0,
        0.5
    )
st.sidebar.title("🎯 AI Object Detection")

st.sidebar.success("✅ YOLOv8 Model Loaded")

st.sidebar.write("### Technologies")
st.sidebar.write("• Python")
st.sidebar.write("• Streamlit")
st.sidebar.write("• OpenCV")
st.sidebar.write("• YOLOv8")

st.sidebar.divider()

st.sidebar.write("👨‍💻 Developed by")
st.sidebar.write("**Sathwika Gattu**")

# Load YOLO model
model = YOLO("yolov8n.pt")
object_metric=st.sidebar.empty()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)

        results = model(temp.name)

    result_image = results[0].plot()

    st.image(result_image, caption="Detected Objects", use_container_width=True)
    from io import BytesIO

    buffer = BytesIO()

    Image.fromarray(result_image).save(buffer, format="PNG")

    st.download_button(
    "📥 Download Result",
    data=buffer.getvalue(),
    file_name="detected_image.png",
    mime="image/png"
)

    st.success("✅ Detection Complete!")
    import cv2

st.divider()
st.subheader("📷 Live Webcam Detection")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while run:
        start=time.time()
        success, frame = cap.read()

        if not success:
            st.error("Cannot access webcam.")
            break

        results = model.track(frame,conf=confidence,persist=True) 
        boxes=results[0].boxes
        names = model.names
        detected = []
        for box in boxes:
            cls = int(box.cls[0]) 
            conf=float(box.conf[0])
            st.write(f"{names[cls]}({conf:.2f})")
            detected.append(names[cls])
            object_metric.metric("object detected",len(detected))
            st.write("### 🏷️ Detected Objects")
            if detected:
                st.write(", ".join(set(detected)))
            else:
                st.write("No objects detected.")
        unique_objects = sorted(set(detected))
        from collections import Counter
        counts = Counter(detected)
        st.subheader("📊 Detection Summary")
        for obj, count in counts.items():
            st.write(f"**{obj}** : {count}")
        st.sidebar.metric("📦 Objects", len(detected))
        st.sidebar.metric("🧾 Unique Classes", len(unique_objects))
        st.write("### 🏷️ Detected Objects")
        if unique_objects:
            st.write(", ".join(unique_objects))
        else:
            st.write("No objects detected.")        

        annotated_frame = results[0].plot(True)

        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )
        end=time.time()
        fps=1/(end-start)
        st.sidebar.metric("FPS",f"{fps:.1f}")

        

    cap.release()