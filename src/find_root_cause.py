import tensorflow as tf
import numpy as np
from pathlib import Path
from src.data_process import Preprocessor,FeatureBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "telecom_model_ml.keras"

class TelecomPipeline:
    def __init__(self, model_path: str, question: str):
        self.question = question
        self.preprocessor = Preprocessor(self.question)
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self: str):
        df = self.preprocessor.build_sequence()
        features = FeatureBuilder(df).build()
        
        if features.shape[1] != 25:
            raise ValueError(f"Feature mismatch: number of columns={features.shape[1]} where it should be 25")

        X = np.expand_dims(features, axis=0)

        preds = self.model.predict(X)

        return {
            "class": int(np.argmax(preds)),
            "probs": preds.tolist()
        }