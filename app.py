import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("face_classifier.h5")
st.title("real vs fake face classifier")
st.write("upload a face image to check if it real or AI generated (fake)")
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
if uploaded_file is not None:
  img = Image.open(uploaded_file).convert("RGB")
  st.image(img,caption="uploaded image", use_column_width=True)

  img_resized= img.resize((128,128))
  img_array= np.expand_dims(np.array(img_resized),axis=0)

  prediction = model.predict(img_array)[0][0]
  label="Real" if prediction > 0.5 else"Fake"
  confidence= prediction if prediction>0.5 else 1- prediction

  st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence*100:.1f}%")
