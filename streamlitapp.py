import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

#To download the h5 model from a google drive link
def download_model():
    url = "https://drive.google.com/file/d/1apEtp090zBptSTY2hPjNcneGikGVvyIi/view?usp=drive_link"
    model_path = "skin_condition_classifier.h5"

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            print("Downloading model...")
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
            print("Download complete.")

download_model()


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