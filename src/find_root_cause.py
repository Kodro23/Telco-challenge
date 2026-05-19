import tensorflow as tf
import numpy as np
from pathlib import Path
from src.preprocessing import Preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"

class TelecomPipeline:
    def __init__(self, model_path: str):
        self.preprocessor = Preprocessor()
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, raw_text: str):
        df = self.preprocessor.build_sequence(raw_text)
        X = df.to_numpy()

        if X.shape[1] != 25:
            raise ValueError("Feature mismatch")

        X = np.expand_dims(X, axis=0)

        preds = self.model.predict(X)

        return {
            "class": int(np.argmax(preds)),
            "probs": preds.tolist()
        }