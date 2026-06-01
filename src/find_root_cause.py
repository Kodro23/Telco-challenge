import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib
from src.data_process import Preprocessor, FeatureBuilder, encode_column

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.pkl"
ROOT_CAUSES = {
    0: ("C1", "The serving cell's downtilt angle is too large, causing weak coverage at the far end."),
    1: ("C2", "The serving cell's coverage distance exceeds 1km, resulting in over-shooting."),
    2: ("C3", "A neighboring cell provides higher throughput."),
    3: ("C4", "Non-colocated co-frequency neighboring cells cause severe overlapping coverage."),
    4: ("C5", "Frequent handovers degrade performance."),
    5: ("C6", "Neighbor cell and serving cell have the same PCI mod 30, leading to interference."),
    6: ("C7", "Test vehicle speed exceeds 40km/h, impacting user throughput."),
    7: ("C8", "Average scheduled RBs are below 160, affecting throughput.")
}

class TelecomPipeline:
    def __init__(self, model_path: str, question: str):
        self.question = question
        self.preprocessor = Preprocessor(self.question)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.encoders = joblib.load(ENCODERS_PATH)
        self.rootcauses=ROOT_CAUSES


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
        pred_class = int(np.argmax(preds))
        label, description = ROOT_CAUSES[pred_class]

        return {
            "Class": label,
            "description": description,
            "probs": preds.tolist()
            }