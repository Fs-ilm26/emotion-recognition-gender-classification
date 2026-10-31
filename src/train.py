import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import EPOCHS, MODEL_PATH
from data_loader import load_datasets
from model import build_multi_task_model, save_history


def train_model():
    train_ds, val_ds = load_datasets()
    model = build_multi_task_model()

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_emotion_output_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    save_history(history, os.path.join(os.path.dirname(MODEL_PATH), "training_history.json"))
    print(f"Training complete. Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an emotion recognition and gender classification model")
    args = parser.parse_args()
    train_model()
