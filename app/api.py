from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.find_root_cause import TelecomPipeline
from pathlib import Path

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"

# -----------------------
# Health check
# -----------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------
# Predict endpoint
# -----------------------
class InputData(BaseModel):
    text: str
@app.post("/predict")
def predict(input: InputData):
    try:
        pipeline = TelecomPipeline(model_path=str(MODEL_PATH), question=input.text)
        return pipeline.predict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))