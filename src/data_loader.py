import os
from typing import Tuple

import cv2
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory

from config import BATCH_SIZE, IMAGE_SIZE, TRAIN_DIR, VALIDATION_DIR, EMOTION_CLASSES


def load_datasets() -> Tuple:
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_ds = image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_ds, val_ds


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMAGE_SIZE)
    frame = frame.astype(np.float32) / 255.0
    return frame


def save_predictions(predictions, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, predictions)
