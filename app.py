import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Real vs Fake Face Classifier", page_icon="🕵️", layout="centered")

model = load_model("face_classifier.h5")

st.title("🕵️ Real vs Fake Face Classifier")
st.write("Upload a face image to check if it's **Real** or **AI-generated (Fake)**.")

st.markdown("---")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        img_resized = img.resize((128, 128))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        prediction = model.predict(img_array)[0][0]

    label = "Real" if prediction > 0.5 else "Fake"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown("### Result")
    if label == "Real":
        st.success(f"✅ Prediction: **{label}**")
    else:
        st.error(f"⚠️ Prediction: **{label}**")

    st.progress(float(confidence))
    st.write(f"Confidence: **{confidence*100:.1f}%**")

    with st.expander("How does this work?"):
        st.write(
            "This app uses a Convolutional Neural Network (CNN) trained on 140,000 "
            "real and AI-generated face images, achieving 94.58% accuracy on unseen test data."
        )

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit")
