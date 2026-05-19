from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from src.find_root_cause import TelecomInference
from src.preprocessing import Preprocessor   

app = FastAPI()

model = None


# -----------------------
# Load model 
# -----------------------
@app.on_event("startup")
def load_model():
    global model
    model = TelecomInference()


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
       
        preprocessor = Preprocessor(question=input.text)
        merged_df = preprocessor.build_sequence()
        X = merged_df.to_numpy()
        # Validity check
        if X.shape[1] != 25:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 25 features, got {X.shape[1]}"
            )

        #Reshape for model
        X = np.expand_dims(X, axis=0)  # (1, T, 25)

        #Predict
        preds = model.predict(X)

        #Output
        return {
            "prediction_class": int(np.argmax(preds)),
            "probabilities": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))