
import streamlit as st
from pathlib import Path
from src.find_root_cause import TelecomPipeline
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parent
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
st.subheader("Submit a telecom log")

question = st.text_area(
    "Telecom log",
    placeholder="Paste your telecom log or JSON request here...",
    height=300
)

analyze = st.button(
    "Analyze log",
    type="primary",
    disabled=not question.strip()
)

# Analysis section
if analyze:
    question_text = question.strip()

    # Accept either raw text or a JSON object containing "text"
    try:
        parsed = json.loads(question_text)

        if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
            question_text = parsed["text"]

    except json.JSONDecodeError:
        pass

    with st.spinner("Analyzing the telecom log..."):
        try:
            pipeline = TelecomPipeline(
                model_path=str(MODEL_PATH),
                question=question_text
            )

            result = pipeline.predict()

            st.subheader("2. Diagnosis results")
            st.success("Analysis completed")

            st.write("**Predicted root cause:**", result["Class"])
            st.write("**Description:**", result["description"])
            st.metric(
                "Model confidence",
                f"{result['confidence']:.1%}"
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
