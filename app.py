import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Authenticity Scanner", page_icon="🔎", layout="centered")

# ---- Custom styling ----
st.markdown("""
<style>
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        border: 1px solid #1E293B;
    }
    .result-real {
        background-color: rgba(45, 212, 191, 0.08);
        border-color: #2DD4BF;
    }
    .result-fake {
        background-color: rgba(248, 113, 113, 0.08);
        border-color: #F87171;
    }
    .verdict-text {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: monospace;
    }
    .verdict-real { color: #2DD4BF; }
    .verdict-fake { color: #F87171; }
    .stat-label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: none;
    }
</style>
""", unsafe_allow_html=True)

model = load_model("face_classifier.h5")

# ---- Header ----
st.markdown('<div class="main-title">🔎 Face Authenticity Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face photo to check whether it is a real photograph or AI-generated.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    img = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(img, caption="Submitted image", use_container_width=True)

    with st.spinner("Scanning pixel patterns..."):
        img_resized = img.resize((128, 128))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        prediction = model.predict(img_array)[0][0]

    label = "REAL" if prediction > 0.5 else "FAKE"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    card_class = "result-real" if label == "REAL" else "result-fake"
    verdict_class = "verdict-real" if label == "REAL" else "verdict-fake"
    icon = "✅" if label == "REAL" else "🚫"

    with col2:
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="stat-label">VERDICT</div>
            <div class="verdict-text {verdict_class}">{icon} {label}</div>
            <br>
            <div class="stat-label">CONFIDENCE</div>
            <div style="font-family: monospace; font-size: 1.4rem;">{confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(confidence))

    with st.expander("How this works"):
        st.write(
            "This scanner uses a Convolutional Neural Network (CNN) trained on 140,000 "
            "real and AI-generated face images, reaching 94.58% accuracy on unseen test data. "
            "It looks for subtle pixel-level patterns — texture, blending artifacts, and lighting "
            "inconsistencies — that AI face generators tend to leave behind."
        )

st.markdown("<br><hr style='border-color:#1E293B'>", unsafe_allow_html=True)
st.caption("Built with TensorFlow & Streamlit · CNN trained from scratch") hey i want to name this as real VS fake image  Classifier
