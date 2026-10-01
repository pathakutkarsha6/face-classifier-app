# 🔎 Real vs Fake Face Classifier

A deep learning project that detects whether a face image is a **real photograph** or **AI-generated (fake)**, built with TensorFlow/Keras and deployed as a live web app using Streamlit.

**🔗 Live App:** [face-classifier-app-e6jz8wphqqsmhaxfdtqw88.streamlit.app](https://face-classifier-app-e6jz8wphqqsmhaxfdtqw88.streamlit.app)

## Overview

This project trains a Convolutional Neural Network (CNN) from scratch to classify face images as Real or Fake, using the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle.

## Tech Stack

- **TensorFlow / Keras** — model building and training
- **Google Colab (GPU)** — training environment
- **Streamlit** — web app interface
- **GitHub + Streamlit Community Cloud** — deployment

## How It Works

1. **Data Pipeline** — 140,000 face images loaded and split into train (100k) / validation (20k) / test (20k) sets
2. **Preprocessing** — images resized to 128x128 and normalized
3. **Data Augmentation** — random flips, rotation, zoom, and brightness adjustments applied during training to improve generalization to varied real-world photos
4. **Model** — a CNN with 3 convolutional blocks followed by dense layers, trained with binary cross-entropy loss
5. **Evaluation** — tested on 20,000 unseen images
6. **Deployment** — trained model served through a Streamlit web interface, allowing real-time predictions on user-uploaded photos

## Results

- **Test Accuracy:** ~-94% depending on configuration
- Model performs best on plain, front-facing, uncropped face photos similar to its training data

## Limitations

This model was trained on a 2020-era dataset of AI-generated faces (StyleGAN). It may not reliably detect images from newer AI generators, heavily stylized or edited photos, or images with unusual formatting (e.g. screenshots with black bars). This is a well-known generalization challenge in deepfake detection research — models trained on one generator's outputs often don't transfer perfectly to other generators or photo styles.

## Future Improvements

- Train on a more diverse dataset including newer AI-generated images
- Experiment with transfer learning (e.g. MobileNet, EfficientNet) for improved accuracy and generalization
- Add face-cropping preprocessing to handle non-standard photo formats

## Run Locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Author

Built by Utkarsha Pathak as a hands-on project to learn deep learning, CNNs, and ML deployment — from data pipeline to trained model to live deployment.
