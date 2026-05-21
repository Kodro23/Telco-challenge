import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib
from src.data_process import Preprocessor, FeatureBuilder, encode_column

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.pkl"


class TelecomPipeline:
    def __init__(self, model_path: str, question: str):
        self.question = question
        self.preprocessor = Preprocessor(self.question)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.encoders = joblib.load(ENCODERS_PATH)

    def predict(self: str):
        df = self.preprocessor.build_sequence()
        categorical_cols = [col for col in df.select_dtypes(include=["object", "string"]).columns if col not in ["ID", "answer", "Timestamp"]]
        for col in categorical_cols:
            df[col] = encode_column(df, col, encoders=self.encoders, training=True)
        features = FeatureBuilder(df).build().values.astype(np.float32)
        if features.shape[1] != 24:
            raise ValueError(f"Feature mismatch: number of columns={features.shape[1]} where it should be 24")

        X = np.expand_dims(features, axis=0)

        preds = self.model.predict(X)

        return {
            "class": int(np.argmax(preds)),
            "probs": preds.tolist()
        }