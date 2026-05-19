from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from src.find_root_cause import TelecomInference
from src.preprocessing import Preprocessor   

app = FastAPI()

model = None


# -----------------------
# Load model once
# -----------------------
@app.on_event("startup")
def load_model():
    global model
    model = TelecomInference()


# -----------------------
# Input = RAW TEXT (IMPORTANT CHANGE)
# -----------------------
class InputData(BaseModel):
    text: str


# -----------------------
# Health check
# -----------------------
@app.get("/")
def health():
    return {"status": "ok"}


# -----------------------
# Predict endpoint
# -----------------------
@app.post("/predict")
def predict(input: InputData):

    try:
        # 1. PREPROCESS TEXT → DATAFRAME / FEATURES
        preprocessor = Preprocessor(question=input.text)

        merged_df = preprocessor.build_sequence()

        # 2. Convert to model input
        # IMPORTANT: your model expects (batch, 10, 25)
        X = merged_df.to_numpy()

        # if your model expects fixed window:
        if X.shape[0] < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough timesteps. Got {X.shape[0]}, expected >= 10"
            )

        # take last 10 timesteps (common in time-series inference)
        X = X[-10:, :25]

        # reshape to (1, 10, 25)
        X = np.expand_dims(X, axis=0)

        # 3. Predict
        preds = model.predict(X)

        # 4. Output
        return {
            "prediction_class": int(np.argmax(preds)),
            "probabilities": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))