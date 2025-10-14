import os
import time
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D  # pyright: ignore[reportMissingImports]

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
np.set_printoptions(suppress=True)


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# Load model and labels
model = load_model(
    "keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
)
class_names = [c.strip() for c in open("labels.txt", "r").readlines()]

camera = cv2.VideoCapture(1)

# Initialize tracking
stable_start_time = None
last_class_index = None
counters = {cls: 0 for cls in class_names}
THRESHOLD = 0.97  # 97%
HOLD_TIME = 2.0   # seconds

while True:
    ret, image = camera.read()
    if not ret:
        break

    # Resize and preprocess
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1

    # Predict
    prediction = model.predict(image_np, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Print to console
    print(f"Class: {class_name} | Confidence: {confidence_score * 100:.2f}%")

    # Stability check for 97%+
    if confidence_score >= THRESHOLD:
        if last_class_index == index:
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= HOLD_TIME:
                counters[class_name] += 1
                print(f"{class_name} counter incremented: {counters[class_name]}")
                stable_start_time = None
        else:
            last_class_index = index
            stable_start_time = time.time()
    else:
        stable_start_time = None
        last_class_index = index

    # Draw counters and info
    y_offset = 40
    cv2.putText(image, f"Class: {class_name}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(image, f"Confidence: {confidence_score * 100:.1f}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 40

    # Show each class counter
    for cls, count in counters.items():
        cv2.putText(image, f"{cls}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if count > 0 else (200, 200, 200), 2)
        y_offset += 30

    cv2.imshow("Webcam Image", image)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
