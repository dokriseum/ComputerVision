import cv2  # OpenCV zur Nutzung der Kamera
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import time

# Modellpfad und Label-Pfad
model_path = '/Users/dokriseum/Projects/CE-Master_Computer_Vision/cv-project-training/VGG16_FALL1/VGG16_model/VGG16_model_2024-09-21_22-50-01/model.h5'  # Pfad zum trainierten Modell
class_names_path = '/Users/dokriseum/Projects/CE-Master_Computer_Vision/cv-project-training/VGG16_FALL1/VGG16_model/VGG16_model_2024-09-21_22-50-01/class_names.json'  # Pfad zur JSON-Datei mit den Klassennamen

# Modell laden
model = load_model(model_path)

# Laden der Klassennamen
with open(class_names_path, 'r') as f:
    class_names = json.load(f)

# Kamera initialisieren
cap = cv2.VideoCapture(0)  # Nutzt die erste Kamera (0 = eingebaute Webcam)

if not cap.isOpened():
    print("Fehler: Konnte die Kamera nicht öffnen.")
    exit()

print("Drücken Sie 'q', um das Programm zu beenden.")

while True:
    # Ein Frame von der Kamera lesen
    ret, frame = cap.read()
    
    if not ret:
        print("Fehler: Konnte das Bild nicht aufnehmen.")
        break

    # Bildgröße ändern auf (128, 128), um dem Modell zu entsprechen
    resized_frame = cv2.resize(frame, (128, 128))

    # Normalisierung
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)  # (1, 128, 128, 3)

    # Vorhersage
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100

    # Ausgabestring erstellen
    label = class_names[str(predicted_class)]
    result_text = f"{label} ({confidence:.2f}%)"

    # Ausgabe des Ergebnisses auf dem Kamerabild
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Kamerabild anzeigen
    cv2.imshow('Live-Bilderkennung', frame)

    # 'q' drücken, um das Programm zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera und Fenster freigeben
cap.release()
cv2.destroyAllWindows()
