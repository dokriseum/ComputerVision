import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
import random


model = tf.keras.models.load_model("best_model.h5")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    frame = cv2.resize(frame, (target_width, target_height))

    preds = model.predict(np.expand_dims(frame, axis=0))
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]

    label = "Cat" if class_idx == 0 else "Dog"
    display_text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Klassifizierung", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

camera.release()
cv2.destroyAllWindows()