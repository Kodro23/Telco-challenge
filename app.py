#Load packages
import streamlit as st
from src.find_root_cause import TelecomPipeline
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"
#Set page configuration
st.set_page_config(
  page_title="Root cause finder",
  page_icon="🔦"
)

#Set title
st.title("Root cause finder")

#Set description
st.write(
  "Enter you telelog and find out what is the root cause"
  "of the network failure"
)

#Get the user's question
question=st.chat_input("Enter your telelog here...")

#Check if the user asked a question
if question:
  #Display the user's question
  with st.chat_message("user"):
    st.write(question)

  #Call the agent
  with st.spinner("Searching..."):
    result= TelecomPipeline(model_path=str(MODEL_PATH), question=input.text)
    
  #Get the final answer
  answer=result.predict()

  #Display the agent's answer
  with st.chat_message("The root cause is:"):
    st.write(answer)