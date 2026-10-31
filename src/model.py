import json
import os

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

from config import EMOTION_CLASSES, GENDER_CLASSES, LEARNING_RATE


def build_multi_task_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    x = base_model.output
    x = layers.Dropout(0.4)(x)

    emotion_logits = layers.Dense(len(EMOTION_CLASSES), activation="softmax", name="emotion_output")(x)
    gender_logits = layers.Dense(len(GENDER_CLASSES), activation="softmax", name="gender_output")(x)

    model = models.Model(inputs=base_model.input, outputs=[emotion_logits, gender_logits])

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            "emotion_output": "sparse_categorical_crossentropy",
            "gender_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "emotion_output": ["accuracy"],
            "gender_output": ["accuracy"],
        },
    )

    return model


def save_history(history, path: str) -> None:
    with open(path, "w") as f:
        json.dump(history.history, f, indent=2)
