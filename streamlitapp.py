import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('skin_condition_classifier.h5')

# Class labels (update with your actual classes)
class_names = ['Acne','Carcinoma','Eczema','Keratosis','Milia', 'Rosacea']

# Streamlit UI
st.title("Skin Condition Classifier")
st.write("Upload an image of a skin condition to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Match training normalization

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"*Prediction:* {predicted_class}")
    st.write(f"*Confidence:* {confidence:.2f}")