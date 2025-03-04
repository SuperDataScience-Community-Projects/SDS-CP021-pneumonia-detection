import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# 1) Load your trained model
MODEL_PATH = r"C:\Users\henry\Desktop\Project\detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# (Optional) Force a dummy forward pass if you like (harmless):
# _ = model.predict(np.zeros((1,64,64,3), dtype=np.float32))

class_labels = ["Normal", "Pneumonia"]

def preprocess_image(img):
    """Resize (64,64), normalize [0,1], expand dims => (1,64,64,3)."""
    img = cv2.resize(img, (64,64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def make_prediction(img):
    """Predict label + confidence from raw image."""
    x = preprocess_image(img)
    pred = model.predict(x)[0][0]
    conf = float(pred if pred > 0.5 else 1 - pred)
    label = class_labels[int(pred > 0.5)]
    return label, conf

st.title("Pneumonia Detection (No Grad-CAM)")

uploaded_file = st.file_uploader("Upload an X-ray", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    # Convert the file into OpenCV BGR format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Make a prediction
    label, confidence = make_prediction(img_rgb)
    st.write(f"### Prediction: {label} ({confidence*100:.2f}%)")
