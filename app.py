import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato.keras')

model = load_model()

# 2. Define Class Names (Update these based on your specific training labels)
CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy'] # Example list

st.title("AgriPathogen Alert: Tomato Disease Detection")
st.write("Upload an image of a tomato leaf to detect potential pathogens.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    # 3. Image Preprocessing (Matches your Training.ipynb logic)
    img = image.resize((256, 256)) # Ensure this matches your model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 4. Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")



