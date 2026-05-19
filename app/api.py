from fastapi import FastAPI,  HTTPException
from pydantic import BaseModel
import numpy as np

from src.find_root_cause import TelecomInference

app = FastAPI(
    root_path="/proxy/8000"
)

model = None  # global placeholder


# ---- Load model safely ----
@app.on_event("startup")
def load_model():
    global model
    model = TelecomInference()


# ---- Input schema ----
class InputData(BaseModel):
    data: list[list[list[float]]]


# ---- API route ----
@app.post("/predict")
from fastapi import HTTPException
import numpy as np

@app.post("/predict")
def predict(input: InputData):

    X = np.array(input.data)

    if X.shape[1:] != (10, 25):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (batch, 10, 25), got {X.shape}"
        )

    preds = model.predict(X)

    return {
        "prediction_class": int(np.argmax(preds)),
        "probabilities": preds.tolist()
    }

    
# ---- Health check (IMPORTANT for debugging) ----
@app.get("/")
def health():
    return {"status": "ok"}