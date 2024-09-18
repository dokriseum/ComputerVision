#! /usr/bin/env python3

# Importe
import os
import sys
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Überprüfung der Argumente und Pfaddefinitionen
if len(sys.argv) != 2:
    print("Fehler: Bitte nur den Pfad zum Projektverzeichnis angeben.")
    sys.exit(1)

projekt_root_pfad = os.path.abspath(sys.argv[1])
daten_root_pfad = os.path.join(projekt_root_pfad, "daten_quadratisch")

# GPU-Einstellungen
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Fehler beim Setzen des Speicherwachstums:", e)

# Präzisionseinstellungen
set_global_policy('mixed_float16')

# Hyperparameter
epochen = 20
batch_groesse = 4
lernrate_start = 0.0001
bild_groesse = (224, 224)
anzahl_klassen = 3

# Daten laden und augmentieren
def zufaellig_grau(image):
    if np.random.rand() < 0.25:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = np.stack((image,) * 3, axis=-1)
    return image

train_gen = ImageDataGenerator(
    preprocessing_function=zufaellig_grau,
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

valid_gen = ImageDataGenerator(
    preprocessing_function=zufaellig_grau,
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.2
)

trainDaten = train_gen.flow_from_directory(
    daten_root_pfad,
    target_size=bild_groesse,
    batch_size=batch_groesse,
    class_mode='categorical',
    subset='training',
    seed=321
)

validDaten = valid_gen.flow_from_directory(
    daten_root_pfad,
    target_size=bild_groesse,
    batch_size=batch_groesse,
    class_mode='categorical',
    subset='validation',
    seed=321
)

# Modell erstellen
basis_modell = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(bild_groesse[0], bild_groesse[0], 3),
)
basis_modell.trainable = False

# Neue Schichten definieren
x = Flatten()(basis_modell.output)
x = Dense(units=256, activation="relu")(x)
ausgabe = Dense(units=anzahl_klassen, activation="softmax")(x)
modell = Model(inputs=basis_modell.inputs, outputs=ausgabe)

# Kompilieren des Modells
modell.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lernrate_start),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
verlauf = modell.fit(
    trainDaten,
    epochs=epochen,
    validation_data=validDaten
)

# Konfusionsmatrix
wahre_labels, vorhergesagte_labels = [], []
for i in range(len(validDaten)):
    x, y = validDaten[i]
    vorhersagen = np.argmax(modell.predict(x), axis=-1)
    wahre_labels.extend(np.argmax(y, axis=-1))
    vorhergesagte_labels.extend(vorhersagen)

konfusionsmatrix = confusion_matrix(wahre_labels, vorhergesagte_labels)
sns.heatmap(konfusionsmatrix, annot=True, fmt="d", cmap="Blues")

# Speichern des Modells
zeitstempel = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
speicherpfad = os.path.join('gespeichertes_modell', f'VGG16_{zeitstempel}')
modell.save(speicherpfad)
