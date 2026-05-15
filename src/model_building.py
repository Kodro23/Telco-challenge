#import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,Input,Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import make_scorer ,confusion_matrix,classification_report
import keras_tuner as kt

#Naive model
def build_model(hp):

    model = Sequential()

    # ✅ Always start with Input layer (IMPORTANT for Keras Tuner stability)
    model.add(Input(shape=(10, 25)))

    # LSTM layer
    model.add(
        LSTM(
            units=hp.Choice("lstm_units", [64, 128,256]),
            return_sequences=False
        )
    )

    # Dropout (kept optional but stable)
    model.add(
        Dropout(
            rate=hp.Choice("dropout", [0.0,0.1, 0.2,0.3,0.4 0.5])
        )
    )

    # Dense layer
    model.add(
        Dense(
            units=hp.Choice("dense_units", [32, 64,128]),
            activation="relu"
        )
    )

    # Output layer
    model.add(Dense(8, activation="softmax"))

    # Learning rate tuning
    lr = hp.Choice("lr", [1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",]
    )

    return model

#Weigted model

#Define F1 score function
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.conf_matrix = self.add_weight(
            name="conf_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes
        )

        self.conf_matrix.assign_add(tf.cast(cm, tf.float32))

    def result(self):
        cm = self.conf_matrix

        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        return tf.reduce_mean(f1)

    def reset_state(self):
        self.conf_matrix.assign(tf.zeros_like(self.conf_matrix))

#Model
def build_model_f1(hp):

    model = Sequential()

    # ✅ Always start with Input layer (IMPORTANT for Keras Tuner stability)
    model.add(Input(shape=(10, 25)))

    # LSTM layer
    model.add(
        LSTM(
            units=hp.Choice("lstm_units", [64, 128,256]),
            return_sequences=False
        )
    )

    # Dropout (kept optional but stable)
    model.add(
        Dropout(
            rate=hp.Choice("dropout", [0.0, 0.1, 0.2,0.3,0.4, 0.5])
        )
    )

    # Dense layer
    model.add(
        Dense(
            units=hp.Choice("dense_units", [32, 64,128]),
            activation="relu"
        )
    )

    # Output layer
    model.add(Dense(8, activation="softmax"))

    # Learning rate tuning
    lr = hp.Choice("lr", [1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",MacroF1(num_classes=8, name="f1")]
    )

    return model