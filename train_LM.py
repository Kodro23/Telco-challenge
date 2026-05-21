# import libraries
# Handle data
import pandas as pd
import numpy as np
# Model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
from src.LM_model_building import build_model
from src.data_process import Preprocessor, FeatureBuilder, encode_column

from pathlib import Path
import joblib
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####################################################################################################
# Let's preprocess data
df = pd.read_csv("./data/raw/train.csv")
processed_rows = []
for idx, row in df.iterrows():
    try:
        processor = Preprocessor(row["question"])
        merged = processor.build_sequence()
        merged["ID"] = row["ID"]
        builder = FeatureBuilder(merged)
        # store tabular version
        processed_rows.append(builder.build())
    except Exception as e:
        print(f"Error on row {idx}: {e}")

# build final dataframe
processed_dataframe = pd.concat(processed_rows, ignore_index=True)
# Label encoding
categorical_encoders = {}
categorical_cols = [col for col in processed_dataframe.select_dtypes(
    include=["object", "string"]).columns if col not in ["ID", "answer", "Timestamp"]]
for col in categorical_cols:
    processed_dataframe[col] = encode_column(processed_dataframe, col, encoders=categorical_encoders, training=True)

# Format data
processed_data = []
for telelog_id in processed_dataframe["ID"].unique():
    # get rows for ONE telelog
    sample_df = processed_dataframe[processed_dataframe["ID"] == telelog_id].copy()
    # sequence = matrix
    sequence = sample_df.drop(columns=["ID", "answer"], errors="ignore").values
    # get label
    label = df[df["ID"] == telelog_id]["answer"].iloc[0]
    processed_data.append({
        "ID": telelog_id,
        "sequence": sequence,
        "label": label
    })

label_map = {f"C{i}": i-1 for i in range(1, 9)}
for sample in processed_data:
    sample["label_id"] = label_map[sample["label"]]

sequences = [
    sample["sequence"]
    for sample in processed_data
]
labels = [
    sample["label_id"]
    for sample in processed_data
]

# Padding
X = pad_sequences(sequences).astype("float32")
# one hot encoding
y = tf.keras.utils.to_categorical(labels, num_classes=8)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
X_test = X_test.astype("float32")
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

####################################################################################################
y_train_labels = np.argmax(y_train, axis=-1).flatten()
classes = np.unique(y_train_labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_labels
)

class_weights = dict(zip(classes, class_weights))

# Training
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_f1", direction="max"),
    max_trials=10,
    directory="tuning",
    project_name="telecom_lstm",
    overwrite=True
)
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights
)
best_model = tuner.get_best_models(num_models=1)[0]
####################################################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Telco-challenge" / "models" / "telecom_model_ml.keras"
ENCODERS_PATH = PROJECT_ROOT / "Telco-challenge" / "models" / "encoders.pkl"
FEATURES_PATH = PROJECT_ROOT / "Telco-challenge" / "models" / "feature_cols.json"

# save model
best_model.save(MODEL_PATH)
# save parameters
joblib.dump(categorical_encoders, ENCODERS_PATH)

feature_cols = list(processed_dataframe.columns)
with open(FEATURES_PATH, "w") as f:
    json.dump(feature_cols, f)
