import tensorflow as tf
import numpy as np
from pathlib import Path
from src.preprocessing import Preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"
#Load model
class TelecomInference:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def predict(self, X):
        """
        X: np.array of shape (batch, 10, 25)
        """
        X = np.array(X)
        preds = self.model.predict(X)
        return preds

class TelecomPipeline:
    def __init__(self):
        self.model = TelecomInference()
        self.preprocessor = Preprocessor()

    def predict(self, text: str):

        preprocessed = self.preprocessor.build_sequence(text)
        X = preprocessed.to_numpy()

        if X.shape[1] != 25:
            raise ValueError(f"Expected 25 features, got {X.shape[1]}")

        X = np.expand_dims(X, axis=0)

        preds = self.model.predict(X)

        return {
            "prediction_class": int(np.argmax(preds)),
            "probabilities": preds.tolist()
        }

