from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.find_root_cause import TelecomPipeline

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT /"Telco-challenge"/ "models" / "telecom_model_ml.keras"
pipeline = TelecomPipeline(model_path=str(MODEL_PATH))



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
        return pipeline.predict(input.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))