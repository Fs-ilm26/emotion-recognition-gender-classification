import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import EMOTION_CLASSES, GENDER_CLASSES
from data_loader import preprocess_frame


def live_demo(model_path: str, camera_index: int = 0):
    model = load_model(model_path)
    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        processed = np.expand_dims(processed, axis=0)
        emotion_probs, gender_probs = model.predict(processed, verbose=0)

        emotion_idx = int(np.argmax(emotion_probs[0]))
        gender_idx = int(np.argmax(gender_probs[0]))

        emotion = EMOTION_CLASSES[emotion_idx]
        gender = GENDER_CLASSES[gender_idx]

        cv2.putText(frame, f"Emotion: {emotion}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Emotion and Gender Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live emotion and gender prediction")
    parser.add_argument("--model", type=str, default="models/emotion_gender_model.keras")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    live_demo(args.model, args.camera)
