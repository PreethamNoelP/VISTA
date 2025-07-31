# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json
from streamlit_lottie import st_lottie
from ultralytics import YOLO

# ------------------------------
# Utility Functions
# ------------------------------

def load_lottiefile(filepath: str):
    """Load Lottie animation from JSON file (optional visuals)."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_model(model_name):
    """Load pre-trained AI models based on user selection."""
    try:
        if model_name == "YOLOv8":
            return YOLO("yolov8n.pt")
        elif model_name == "ResNet50":
            return tf.keras.applications.ResNet50(weights='imagenet')
        elif model_name == "MobileNetV2":
            return tf.keras.applications.MobileNetV2(weights='imagenet')
        elif model_name == "VGG16":
            return tf.keras.applications.VGG16(weights='imagenet')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    return None

def preprocess_image(image):
    """Resize and normalize image for AI model."""
    try:
        image = image.convert('RGB').resize((224, 224))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing error: {str(e)}")
        return None

def detect_faces(image):
    """Detect faces in an image using OpenCV."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img_array
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        return None

def apply_edge_detection(image):
    """Apply Canny edge detection to an image."""
    try:
        img_array = np.array(image.convert('L'))
        return cv2.Canny(img_array, 100, 200)
    except Exception as e:
        st.error(f"Edge detection error: {str(e)}")
        return None

def apply_blurring(image):
    """Apply Gaussian blur to an image."""
    try:
        img_array = np.array(image.convert('RGB'))
        return cv2.GaussianBlur(img_array, (15, 15), 0)
    except Exception as e:
        st.error(f"Blurring error: {str(e)}")
        return None

def compute_histogram(image):
    """Compute grayscale histogram of an image."""
    try:
        img_array = np.array(image.convert('L'))
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        return hist
    except Exception as e:
        st.error(f"Histogram computation error: {str(e)}")
        return None

def object_detection(image, model):
    """Perform object detection using YOLO model."""
    try:
        img_array = np.array(image.convert('RGB'))
        results = model(img_array)
        return results[0].plot()
    except Exception as e:
        st.error(f"Object detection error: {str(e)}")
        return None

def summarize_video(video_path):
    """Placeholder function for video summarization."""
    return "Summary: Detected objects, motion trends, scene changes, and major actions in video."

def process_video(video_path, feature_option, model):
    """Process video frame by frame with selected features."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if "Object Detection" in feature_option:
            results = model(frame)
            frame = results[0].plot()

        if "Face Detection" in feature_option:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if "Edge Detection" in feature_option:
            frame = cv2.Canny(frame, 100, 200)

        if "Blurring" in feature_option:
            frame = cv2.GaussianBlur(frame, (15, 15), 0)

        if "Histogram" in feature_option:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            st.line_chart(hist.flatten())

        st.image(frame, channels="BGR", use_container_width=True)
    cap.release()

# ------------------------------
# Main App
# ------------------------------

def main():
    st.title("VISTA – Vision-based Intelligent System")

    with st.sidebar:
        st.header("Options")
        option = st.selectbox("Choose Analysis Type", ["Image Analysis", "Video Analysis"])
        model_option = st.selectbox("Select AI Model", ["YOLOv8", "ResNet50", "MobileNetV2", "VGG16"])
        feature_option = st.multiselect(
            "Additional Features",
            ["Object Detection", "Edge Detection", "Face Detection", "Blurring", "Histogram"],
            default=["Object Detection"]
        )
        model = load_model(model_option)

    if option == "Image Analysis":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            for feature in feature_option:
                if feature == "Object Detection" and model:
                    st.image(object_detection(image, model), caption="Object Detection", use_container_width=True)
                elif feature == "Face Detection":
                    st.image(detect_faces(image), caption="Face Detection", use_container_width=True)
                elif feature == "Edge Detection":
                    st.image(apply_edge_detection(image), caption="Edge Detection", use_container_width=True)
                elif feature == "Blurring":
                    st.image(apply_blurring(image), caption="Blurred Image", use_container_width=True)
                elif feature == "Histogram":
                    st.line_chart(compute_histogram(image))

    elif option == "Video Analysis":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            st.video(uploaded_video)
            process_video(uploaded_video.name, feature_option, model)
            st.write(summarize_video(uploaded_video.name))

if __name__ == "__main__":
    main()
