import src.LM_model_building
import src.data_process

#Format data
processed_data = []
for telelog_id in processed_dataframe["ID"].unique():
    # get rows for ONE telelog
    sample_df = processed_dataframe[
        processed_dataframe["ID"] == telelog_id
    ].copy()

    # sort by time
    sample_df = sample_df.sort_values("Timestamp").drop(columns=["Timestamp"])
    for col in sample_df.columns[sample_df.columns.isin(sample_df.select_dtypes(include=["object", "string"]).columns)]:
        le = LabelEncoder()
        sample_df[col] = le.fit_transform(sample_df[col])

    # sequence = matrix
    sequence = sample_df.values

    # get label
    label = df[df["ID"] == telelog_id]["answer"].iloc[0]

    processed_data.append({
        "ID": telelog_id,
        "sequence": sequence,
        "label": label
    })

label_map = {
    "C1":0,
    "C2":1,
    "C3":2,
    "C4":3,
    "C5":4,
    "C6":5,
    "C7":6,
    "C8":7}
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
#one hot encoding
y = to_categorical(labels, num_classes=8)
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
X_test = X_test.astype("float32")
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

a
#Training
tuner = kt.RandomSearch(
    build_model,
    objective = kt.Objective("val_f1", direction="max"),
    max_trials=10,
    directory="tuning",
    project_name="telecom_lstm",
    overwrite=True
)
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight= class_weights
)
best_model = tuner.get_best_models(num_models=1)[0]