import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Load model
model = tf.keras.models.load_model("models/catdog_cnn.h5")

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Determine label
    if prediction >= 0.5:
        label = "Dog"
    else:
        label = "Cat"

    # Confidence %
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    confidence = confidence * 100

    st.write(f"### Prediction: **{label}**")
    st.write(f"### Confidence: **{confidence:.2f}%**")

