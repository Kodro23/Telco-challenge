
import streamlit as st
from pathlib import Path
from src.find_root_cause import TelecomPipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"

st.set_page_config(
    page_title="Root Cause Finder",
    page_icon="🔦",
    layout="centered"
)

st.title("🔦 Root Cause Finder")
st.write(
    "Analyze telecom logs to identify the likely cause "
    "of a network failure."
)

# Input section
st.subheader("1. Submit a telecom log")

question = st.text_area(
    "Telecom log",
    placeholder="Paste your telecom log here...",
    height=250
)

analyze = st.button(
    "Analyze log",
    type="primary",
    disabled=not question.strip()
)

# Analysis section
if analyze:
    with st.spinner("Analyzing the telecom log..."):
        try:
            pipeline = TelecomPipeline(
                model_path=str(MODEL_PATH),
                question=question
            )
            result = pipeline.predict()

            st.subheader("2. Diagnosis results")
            st.success("Analysis completed")
            st.write(result)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
