import os
import sys
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt   
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Activation, Dropout, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


if len(sys.argv) != 2:
    print("Argument Error: Please provide just the root project path as argument.")
    sys.exit(1)
project_root_path = os.path.abspath(sys.argv[1])
data_root_path = os.path.join(project_root_path, "dataset_quadratisch_128")

if not os.path.exists(data_root_path):
    raise FileNotFoundError(f"Dataset folder not found: {data_root_path}")

# GPU 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


EPOCHS = 25
BATCH_SIZE = 4
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 3  # (bike, bin, shield)
INITIAL_LEARNING_RATE = 0.00001
SPLIT = 0.2
AUTOTUNE = tf.data.AUTOTUNE


# Fortgeschrittene Data Augmentation Einstellungen
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # Normalisieren der Pixelwerte
    rotation_range=60,                  # Erhöht die Spanne der zufälligen Rotation
    width_shift_range=0.3,              # Erhöht die Spanne für horizontale Verschiebungen
    height_shift_range=0.3,             # Erhöht die Spanne für vertikale Verschiebungen
    shear_range=0.35,                   # Erhöht die Scherintensität
    zoom_range=[0.7, 1.3],              # Ändert den Bereich für zufälliges Zoomen
    channel_shift_range=60.0,           # Intensität der Farbkanalverschiebung
    horizontal_flip=True,               # Horizontales Flippen zulassen
    vertical_flip=True,                 # Vertikales Flippen zulassen
    brightness_range=[0.4, 1.6],        # Erweitert den Bereich für zufällige Helligkeitsänderungen
    fill_mode='wrap',                   # Ändert die Füllmethode für neu erschaffene Pixel
    validation_split=0.2                # Teilt den Datensatz in Trainings- und Validierungsdaten
)

# Setup für Trainings- und Validierungsgeneratoren
train_generator = train_datagen.flow_from_directory(
    data_root_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    data_root_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# Xception model
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = False

input = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model(input, training=False)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
saved_model_dir = os.path.join(project_root_path, "XCEPTION_model", f"XCEPTION_model_{timestamp}")
os.makedirs(saved_model_dir, exist_ok=True)



history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
)


plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('F3 Training and Validation Loss')
plt.savefig(os.path.join(saved_model_dir, 'training_validation_loss.png'), format='png')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('F3 Training and Validation Accuracy')
plt.savefig(os.path.join(saved_model_dir, 'training_validation_accuracy.png'), format='png')
plt.show()



# Modell speichern mit der richtigen Dateierweiterung
model_save_path = os.path.join(saved_model_dir, 'model.h5')  # Verwende .h5 für das HDF5-Format
model.save(model_save_path)
print(f"Modell gespeichert unter: {model_save_path}")



class_names = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(saved_model_dir, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)

print(f"Model and class names saved to {saved_model_dir}")

# 예측 및 Confusion Matrix 계산

# Vorhersagen für das Validierungsset
validation_generator.reset()  # Stellt sicher, dass die Reihenfolge der Vorhersagen und Labels übereinstimmt
predictions = model.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)  # Konvertiere die Wahrscheinlichkeiten zu Klassenindizes

# Tatsächliche Labels
true_classes = validation_generator.classes

# Erstellen der Konfusionsmatrix
cm = confusion_matrix(true_classes, predicted_classes)

# Visualisieren der Konfusionsmatrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Vorhergesagt')
plt.ylabel('Wahr')
plt.title('Konfusionsmatrix')
conf_matrix_path = os.path.join(saved_model_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.show()

# Ausgeben des Speicherorts der Konfusionsmatrix
print(f"Konfusionsmatrix gespeichert unter: {conf_matrix_path}")