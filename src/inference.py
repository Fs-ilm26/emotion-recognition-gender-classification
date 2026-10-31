import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import EMOTION_CLASSES, GENDER_CLASSES
from data_loader import preprocess_frame


def predict_image(model_path: str, image_path: str):
    model = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed = preprocess_frame(image)
    processed = np.expand_dims(processed, axis=0)
    emotion_prob, gender_prob = model.predict(processed, verbose=0)

    emotion_index = int(np.argmax(emotion_prob[0]))
    gender_index = int(np.argmax(gender_prob[0]))

    emotion = EMOTION_CLASSES[emotion_index]
    gender = GENDER_CLASSES[gender_index]
    print(f"Emotion: {emotion}")
    print(f"Gender: {gender}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion and gender prediction on a static image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/emotion_gender_model.keras")
    args = parser.parse_args()
    predict_image(args.model, args.image)
