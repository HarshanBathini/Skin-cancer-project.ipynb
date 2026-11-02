"""Streamlit demo: upload a skin lesion photo and get predicted class probabilities.
Run:  streamlit run app.py   (after train.py has created outputs/skin_cancer_model.keras)
"""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
HIGH_RISK = {"mel", "bcc", "akiec"}


@st.cache_resource
def load():
    model = tf.keras.models.load_model("outputs/skin_cancer_model.keras")
    meta = json.load(open("outputs/classes.json"))
    return model, meta["classes"], meta["names"]


st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺")
st.title("🩺 Skin Cancer Prediction")
st.warning("Educational project only. This is NOT a medical diagnosis. "
           "See a dermatologist for any concerning skin lesion.")

model, classes, names = load()
file = st.file_uploader("Upload a close-up image of the skin lesion", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32")[None]
    probs = model.predict(arr)[0]
    order = np.argsort(probs)[::-1]

    top = classes[order[0]]
    st.subheader(f"Prediction: {names[top]} ({probs[order[0]]:.1%})")
    if top in HIGH_RISK:
        st.error("This class is potentially malignant / pre-cancerous. Please consult a doctor.")
    else:
        st.success("This class is usually benign, but only a doctor can confirm.")

    st.bar_chart({names[classes[i]]: float(probs[i]) for i in order})
