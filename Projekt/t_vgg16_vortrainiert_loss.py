#! /usr/bin/env python3

# Importe
import os
import sys
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, applications, layers, optimizers, preprocessing
from tensorflow.keras.utils import set_global_policy
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Parameter prüfen und Pfade definieren
if len(sys.argv) != 2:
    print("Fehler: Bitte genau einen Pfad zum Projektwurzelverzeichnis angeben.")
    sys.exit(1)

wurzel_pfad = os.path.abspath(sys.argv[1])
daten_pfad = os.path.join(wurzel_pfad, "daten_quadratisch")

# GPU-Einstellungen anpassen
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Speicherwachstum konnte nicht gesetzt werden:", e)

# Präzisionseinstellungen für das Modell
set_global_policy('mixed_float16')

# Hyperparameter festlegen
EPOCHEN = 40
BATCH_GROESSE = 4
START_LERNRATE = 0.0001
BILD_GROESSE = (224, 224)
KLASSEN_ANZAHL = 3

# Datenverarbeitung und -erweiterung
def zufaellig_grau_konvertieren(bild):
    if np.random.rand() < 0.25:
        graubild = np.dot(bild[..., :3], [0.299, 0.587, 0.114])
        return np.stack((graubild,) * 3, axis=-1)
    return bild

train_datengenerator = preprocessing.image.ImageDataGenerator(
    preprocessing_function=zufaellig_grau_konvertieren,
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

valid_datengenerator = preprocessing.image.ImageDataGenerator(
    preprocessing_function=zufaellig_grau_konvertieren,
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.2
)

train_daten = train_datengenerator.flow_from_directory(
    daten_pfad,
    target_size=BILD_GROESSE,
    batch_size=BATCH_GROESSE,
    class_mode='categorical',
    subset='training',
    seed=321
)

valid_daten = valid_datengenerator.flow_from_directory(
    daten_pfad,
    target_size=BILD_GROESSE,
    batch_size=BATCH_GROESSE,
    class_mode='categorical',
    subset='validation',
    seed=321
)

# Modell erstellen und konfigurieren
basis_modell = applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(BILD_GROESSE[0], BILD_GROESSE[0], 3),
)
basis_modell.trainable = False

eingabe = layers.Flatten()(basis_modell.output)
eingabe = layers.Dense(units=256, activation="relu")(eingabe)
ausgabe = layers.Dense(units=KLASSEN_ANZAHL, activation="softmax")(eingabe)
modell = Model(inputs=basis_modell.inputs, outputs=ausgabe)

kosinus_abfall = optimizers.schedules.CosineDecay(
    initial_learning_rate=START_LERNRATE,
    decay_steps=(train_daten.n // BATCH_GROESSE) * EPOCHEN
)

modell.compile(
    optimizer=optimizers.Adam(learning_rate=kosinus_abfall),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training durchführen
verlauf = modell.fit(
    train_daten,
    epochs=EPOCHEN,
    validation_data=valid_daten
)

# Konfusionsmatrix und Ergebnisse darstellen
wahr_labels, vorhersage_labels = [], []
for i in range(len(valid_daten)):
    x, y = valid_daten[i]
    vorhersagen = np.argmax(modell.predict(x), axis=-1)
    wahr_labels.extend(np.argmax(y, axis=-1))
    vorhersage_labels.extend(vorhersagen)

konfusionsmatrix = confusion_matrix(wahr_labels, vorhersage_labels)
sns.heatmap(konfusionsmatrix, annot=True, fmt="d", cmap="Blues")

# Trainingsergebnisse speichern und visualisieren
zeitstempel = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
speicher_pfad = os.path.join('modelle', f'VGG16_{zeitstempel}')
modell.save(speicher_pfad)

klasse_namen = {v: k for k, v in train_daten.class_indices.items()}
with open(os.path.join('modelle', f'klassen_{zeitstempel}.json'), 'w') as f:
    json.dump(klasse_namen, f)
