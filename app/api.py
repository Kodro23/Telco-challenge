from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.find_root_cause import TelecomInference

app = FastAPI()

model = None  # global placeholder


# ---- Load model safely ----
@app.on_event("startup")
def load_model():
    global model
    model = TelecomInference()


# ---- Input schema ----
class InputData(BaseModel):
    data: list


# ---- API route ----
@app.post("/predict")
def predict(input: InputData):

    X = np.array(input.data).reshape(1, 10, 25)

    preds = model.predict(X)

    return {
        "prediction_class": int(np.argmax(preds)),
        "probabilities": preds.tolist()
    }


# ---- Health check (IMPORTANT for debugging) ----
@app.get("/")
def health():
    return {"status": "ok"}