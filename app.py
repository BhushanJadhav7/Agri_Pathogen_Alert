import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

st.set_page_config(page_title="AgriPathogen Alert", page_icon=":tada:", layout="wide")

#header
with st.container():
    st.title("AgriPathogen Alert :tomato:")
    st.write("A tomato disease prediction system using machine learning (ML) is a system that uses algorithms to analyze data from potato crops and predict the likelihood of certain diseases occurring. The system uses various ML techniques, such as decision trees, neural networks, or support vector machines, to analyze the data and make predictions. These predictions can then be used to inform farmers and agricultural professionals on how to best protect their crops and prevent disease outbreaks.")

with st.container():
    st.write("---")
    st.subheader("Enter the image of tomato leaf:")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    # Divide the screen into two columns
    col1, col2 = st.columns(2)
    # Display the image using Streamlit
    col1.image(image, caption='Uploaded Image')

try:
    model = tf.keras.models.load_model('tomato.keras')
    st.write("Model loaded successfully!")
except:
    st.write("Error: Failed to load model.")

predict_button = st.button("PREDICT")

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    class_names = ['Tomato__YellowLeaf__Curl__Virus',
                    'Tomato__Bacterial__Spot',
                    'Tomato__Healthy',
                    'Tomato__Mosaic__Virus']
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if predict_button:
    predicted_class, confidence = predict(model, image)
    st.write("Predicted: ", predicted_class)
    st.write("Confidence: ", confidence)



