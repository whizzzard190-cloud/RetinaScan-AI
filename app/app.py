import sys
import os

# Add project root to Python path FIRST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile
import tensorflow as tf

from models.model_loader import DRPredictor
from gradcam.gradcam_utils import make_gradcam_heatmap, save_gradcam
from preprocessing.preprocess import preprocess_image
from reports.report_generator import generate_pdf

st.set_page_config(page_title="RetinaScan-AI", layout="centered")

st.title("🩺 RetinaScan-AI")
st.subheader("Deterministic Diabetic Retinopathy Detection")

predictor = DRPredictor()

uploaded = st.file_uploader("Upload Retinal Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        image_path = tmp.name

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing retina..."):

        # Prediction (with deterministic cache)
        result = predictor.predict(image_path)

        # Grad-CAM
        model = tf.keras.models.load_model("models/dr_model.h5")
        img = preprocess_image(image_path)
        heatmap = make_gradcam_heatmap(img, model)
        heatmap_path = save_gradcam(image_path, heatmap)

    st.success("Analysis Complete")

    st.markdown("## 🧠 Diagnosis Result")

    st.write(f"**DR Level:** {result['class']}")
    st.write(f"**Confidence:** {result['confidence']} %")

    if result["cached"]:
        st.info("Result loaded from deterministic cache")

    st.markdown("## 🔍 Explainable Heatmap")
    st.image(heatmap_path)

    st.markdown("## 📄 Auto Medical Report")

    pdf_path = generate_pdf(image_path, heatmap_path, result)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Medical Report (PDF)",
            data=f,
            file_name="RetinaScan_Report.pdf",
            mime="application/pdf"
        )