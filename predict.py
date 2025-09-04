import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from task import Net, download_kaggle_dataset

# -----------------------------
# Load PyTorch Model
# -----------------------------
def load_model(path, num_classes):
    model = Net(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# -----------------------------
# Predict Emotion
# -----------------------------
def predict_image(model, image, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, 1).item()

    predicted_class = class_names[predicted_idx]
    confidence = probs[0][predicted_idx].item() * 100
    return predicted_class, confidence

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🖼️ Emotion Detection from Images & Camera")
    st.write("Upload an image or capture one from your camera for emotion prediction.")

    # Load dataset for class names
    dataset_path = download_kaggle_dataset()
    dataset = datasets.ImageFolder(root=dataset_path)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Load latest PyTorch model
    model_files = [f for f in os.listdir(".") if f.startswith("saved_model_round")]
    latest_model = sorted(model_files)[-1] if model_files else "saved_model.pth"
    st.sidebar.info(f"Using model: {latest_model}")
    emotion_model = load_model(latest_model, num_classes)

    # -----------------------------
    # Image Upload (keep as-is)
    # -----------------------------
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predicted_class, confidence = predict_image(emotion_model, image, class_names)
        st.success(f"### 🎯 Predicted Emotion: **{predicted_class}** ({confidence:.2f}%)")

    # -----------------------------
    # Camera Capture
    # -----------------------------
    st.write("---")
    st.subheader("📷 Capture Image from Camera")
    start_camera = st.checkbox("Start Camera")
    capture_btn = st.button("Capture & Predict Emotion")
    frame_placeholder = st.empty()  # to show live feed

    cap = None
    captured_frame = None
    if start_camera:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                captured_frame = frame_rgb

    if capture_btn and captured_frame is not None:
        # Load YOLO model
        yolo = YOLO("yolov8n.pt")
        results = yolo(np.array(captured_frame))

        person_detected = False
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label_name = yolo.names[cls_id].lower()
                if label_name in ["person", "face"]:  # adjust based on your YOLO training
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_region = captured_frame[y1:y2, x1:x2]
                    pil_face = Image.fromarray(detected_region)
                    predicted_class, confidence = predict_image(emotion_model, pil_face, class_names)
                    st.success(f"🎯 Predicted Emotion: **{predicted_class}** ({confidence:.2f}%)")
                    break  # take first detected person

        if not person_detected:
            st.warning("No person detected. Try again.")

    if cap:
        cap.release()

if __name__ == "__main__":
    main()
