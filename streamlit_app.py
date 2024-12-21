import streamlit as st
import cv2
import torch
import tempfile
from reportlab.pdfgen import canvas
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(video_path):
    """
    Detect objects in the uploaded video using YOLOv5.
    """
    cap = cv2.VideoCapture(video_path)
    objects_detected = set()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)  # Object detection
        detected = results.pandas().xyxy[0]['name'].tolist()
        objects_detected.update(detected)
    cap.release()
    return list(objects_detected)

def generate_report(objects, compass_data, output_path):
    """
    Generate a summary report as a PDF.
    """
    pdf = canvas.Canvas(output_path)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 750, "Home Object Detection Report")
    pdf.drawString(50, 720, f"Compass Direction: {compass_data} degrees")
    pdf.drawString(50, 700, "Detected Objects:")
    y = 680
    for obj in objects:
        pdf.drawString(70, y, f"- {obj}")
        y -= 20
    pdf.save()

# Streamlit UI
st.title("AR Video Recorder and Object Detection")
st.markdown("""
This application lets you upload a video, analyze it for object detection, and generate a report with detected objects and compass data.
""")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
compass_direction = st.text_input("Compass Direction (degrees)", "Enter compass data manually.")

if video_file:
    # Display video
    st.video(video_file)
    
    # Detect objects
    with st.spinner("Analyzing video for object detection..."):
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(video_file.read())
            detected_objects = detect_objects(temp_video.name)
    
    st.success("Object detection complete!")
    st.markdown("### Detected Objects:")
    for obj in detected_objects:
        st.write(f"- {obj}")

    # Generate report
    if st.button("Generate Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            generate_report(detected_objects, compass_direction, temp_pdf.name)
            st.success("Report generated successfully!")
            st.download_button("Download Report", temp_pdf.name, "report.pdf")

# Compass Integration (Simulated for Web-Based Apps)
st.markdown("""
### Compass Note:
For real-time compass data, consider integrating with a mobile-friendly API using JavaScript and send the data back to Streamlit.
""")
