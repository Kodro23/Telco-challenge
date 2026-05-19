import tensorflow as tf
import numpy as np
from pathlib import Path

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

# model = TelecomInference()

# #Predict
# file_path = sys.argv[1]
# X = np.load(file_path)
# preds = model.predict(X)
# print(preds.argmax(axis=1))