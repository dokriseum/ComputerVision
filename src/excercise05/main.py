import labels
import predictions
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import cv2
import os
import random

# Basispfad zum Datensatz
base_dir = '/Users/dokriseum/Project/CE-M_Computer_Vision/cvss-24-gruppe-1/src/excercise05/cats_vs_dogs'

# Bildgeneratoren vorbereiten mit Datenaufteilung für Training und Validierung
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% der Daten für Validierung
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',  # setzt subset auf 'training'
    classes=['cats', 'dogs']  # Namen der Unterordner
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation',  # setzt subset auf 'validation'
    classes=['cats', 'dogs']  # Namen der Unterordner
)

print("Trainingsbilder:", train_generator.samples)
print("Validierungsbilder:", validation_generator.samples)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Trainingsschleife
for images, labels in train_generator:
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # hier Zugriff auf die Shape der Labels und Vorhersagen
    print("Labels shape:", labels.shape)
    print("Predictions shape:", predictions.shape)

# ggf. Validierungsschleife
for images, labels in validation_generator:
    predictions = model(images, training=False)
    print("Validation Labels shape:", labels.shape)
    print("Validation Predictions shape:", predictions.shape)

print("Labels shape:", labels.shape)
print("Predictions shape:", predictions.shape)

if len(labels) > 0 and len(predictions) > 0:
    loss = tf.keras.losses.binary_crossentropy(labels, predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
best_acc = 0.0
for epoch in range(epochs):
    random = random

    # Training Loop
    for images, labels in train_generator:
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = tf.keras.losses.binary_crossentropy(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Validation Loop
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"Epoch {epoch}: Val Loss: {val_loss}, Val Acc: {val_acc}")

    # bestes Model speichern
    if val_acc > best_acc:
        best_acc = val_acc
        model.save('best_model.h5')

model = tf.keras.models.load_model('best_model.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bild vorbereiten
    resized_frame = cv2.resize(frame, (150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_frame)
    img_array = tf.expand_dims(img_array, 0)  # Erstellt eine Batch

    predictions = model.predict(img_array)
    label = 'Dog' if predictions[0] > 0.5 else 'Cat'

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
