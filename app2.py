import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2

st.set_page_config(page_title="AgriPathogen Alert", page_icon=":tomato:", layout="wide")

# Header
with st.container():
    st.title("AgriPathogen Alert 🍅")
    st.write("A tomato disease prediction system using machine learning (ML).")

with st.container():
    st.write("---")
    st.subheader("Upload an image of a tomato leaf:")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(img, caption='Uploaded Image', use_column_width=True)

try:
    model = load_model('AgriPathogen_Improved.h5')
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.write(f"❌ Error: Failed to load model. {e}")

predict_button = st.button("PREDICT")

def predict(model, img):
    img = img.resize((224, 224))  # Resize for EfficientNet
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    
    class_names = [
        'Tomato__Healthy',
        'Tomato__Bacterial__Spot',
        'Tomato__Mosaic__Virus',
        'Tomato__YellowLeaf__Curl__Virus'
    ]
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

if predict_button and uploaded_file is not None:
    predicted_class, confidence = predict(model, img)
    st.success(f"Predicted: **{predicted_class}**")
    st.info(f"Confidence: **{confidence}%**")
