#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from datetime import datetime
from sklearn.metrics import confusion_matrix

if len(sys.argv) != 2:
    print("Fehler: Bitte geben Sie genau einen Pfad als Argument an.")
    sys.exit(1)

projektPfad = os.path.abspath(sys.argv[1])
datenPfad = os.path.join(projektPfad, "daten_quadratisch")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

Epochen = 20
BatchGroesse = 8
StartLernrate = 0.0001  # Reduzierte Lernrate
BildGroesse = (224, 224)
KlassenAnzahl = 3

def zufallGraustufen(image):
    if np.random.rand() < 0.25:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        image = np.stack((image,) * 3, axis=-1)
    return image

trainGenerator = ImageDataGenerator(
    preprocessing_function=zufallGraustufen,
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

validierungsGenerator = ImageDataGenerator(
    preprocessing_function=zufallGraustufen,
    rescale=1./255,
    featurewise_center=True, 
    featurewise_std_normalization=True,
    validation_split=0.2
)

trainDaten = trainGenerator.flow_from_directory(
    datenPfad,
    target_size=BildGroesse,
    batch_size=BatchGroesse,
    class_mode='categorical',
    subset='training',
    seed=321
)

validierungsDaten = validierungsGenerator.flow_from_directory(
    datenPfad,
    target_size=BildGroesse,
    batch_size=BatchGroesse,
    class_mode='categorical',
    subset='validation',
    seed=321 
)

# Fit generator to calculate statistics for featurewise normalization
trainDaten.reset()  # Ensure we are at the start
sample_images, _ = next(trainDaten)  # Get a batch of images
trainGenerator.fit(sample_images)

Eingabe = Input(shape=(BildGroesse[0], BildGroesse[0], 3))
x = Conv2D(64, 3, padding='same', activation='relu')(Eingabe)
x = BatchNormalization()(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = MaxPool2D(2, strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
Ausgabe = Dense(KlassenAnzahl, activation='softmax')(x)

Modell = Model(inputs=Eingabe, outputs=Ausgabe)
Modell.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Modell.fit(
    trainDaten,
    epochs=Epochen,
    steps_per_epoch=trainDaten.samples // BatchGroesse,
    validation_data=validierungsDaten,
    validation_steps=validierungsDaten.samples // BatchGroesse
)

wahreLabels = []
vorhergesagteLabels = []
for i in range(len(validierungsDaten)):
    batch = validierungsDaten[i]
    vorhergesagteLabels.extend(np.argmax(Modell.predict(batch[0]), axis=-1))
    wahreLabels.extend(np.argmax(batch[1], axis=-1))

konfusionsMatrix = confusion_matrix(np.array(wahreLabels), np.array(vorhergesagteLabels))
print("Konfusionsmatrix:")
print(konfusionsMatrix)

plt.figure(figsize=(8, 6))
sns.heatmap(konfusionsMatrix, annot=True, fmt="d", cmap="Blues")
plt.title("Konfusionsmatrix")
plt.xlabel("Vorhergesagte Klassen")
plt.ylabel("Wahre Klassen")
plt.show()

modellVerzeichnis = os.path.join('modell_gespeichert', f'Modell_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.h5')
os.makedirs(os.path.dirname(modellVerzeichnis), exist_ok=True)
Modell.save(modellVerzeichnis)

klassenIndizes = trainDaten.class_indices
klassenNamen = {v: k for k, v in klassenIndizes.items()}
with open(os.path.join('modell_gespeichert', 'klassenNamen.json'), 'w') as f:
    json.dump(klassenNamen, f)
