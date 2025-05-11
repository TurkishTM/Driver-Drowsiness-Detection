# app.py
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model(r'DDD_model.h5')

# Define class labels (same as training)
CLASSES = {0: "Closed", 1: "Open", 2: "no_yawn", 3: "yawn"}

# Streamlit UI
st.title("Drowsiness Detection 🥱")
st.write("Upload an image of a face to check for drowsiness.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image (match training steps)
    img_array = np.array(image)
    resized = cv2.resize(img_array, (150, 150))  # Resize to 150x150
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    gray = np.expand_dims(gray, axis=-1)  # Add channel dimension (150,150,1)
    gray = np.expand_dims(gray, axis=0)  # Add batch dimension (1,150,150,1)
    gray = gray / 255.0  # Normalize pixels (critical!)
    
    # Predict
    prediction = model.predict(gray)
    class_idx = np.argmax(prediction, axis=1)[0]
    label = CLASSES[class_idx]
    
    # Show result
    st.subheader(f"Prediction: *{label}*")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")