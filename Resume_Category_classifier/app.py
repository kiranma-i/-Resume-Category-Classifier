"""Streamlit app for Resume Category Classification."""

from pathlib import Path

import joblib
import streamlit as st

# Load the saved model from the project folder
MODEL_PATH = Path(__file__).parent / "resume_category_model.pkl"

st.set_page_config(
    page_title="Resume Category Classifier",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Resume Category Classifier")
st.write(
    "Paste a resume below and predict its job category "
    "using a traditional machine learning model."
)

# Check whether the model exists
if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.info("Make sure resume_category_model.pkl is in the project folder.")
    st.stop()

# Load the trained pipeline
pipeline = joblib.load(MODEL_PATH)

st.caption("Model: TF-IDF + Linear SVM")

with st.expander("About this project"):
    st.write("This project classifies resumes into job categories using machine learning.")
    st.write(
        "Predictions are based on patterns in the training dataset "
        "and should not be used as automatic hiring decisions."
    )

# Get resume text from the user
resume_text = st.text_area(
    "Resume text",
    height=260,
    placeholder="Paste resume summary, skills, education, projects, and experience..."
)

# Predict the category
if st.button("Predict Category", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.warning("Please paste resume text first.")
    else:
        prediction = pipeline.predict([resume_text.strip()])[0]
        st.success(f"Predicted category: {prediction}")

st.divider()
st.caption("Built with Python, scikit-learn, TF-IDF, and Streamlit.")