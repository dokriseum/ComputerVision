#import os
#import tensorflow as tf
#from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Basispfad zum Datensatz
# base_dir = '/Users/dokriseum/Project/CE-M_Computer_Vision/cvss-24-gruppe-1/src/excercise05/cats_vs_dogs'

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
import random

def resize_and_pad_images(image_paths, target_width, target_height):
    processed_images = []
    for path in image_paths:
        img = cv2.imread(path)
        height, width, _ = img.shape
        aspect_ratio = width / height

        if 0.8 <= aspect_ratio <= 1.2:
            resized_img = cv2.resize(img, (target_width, target_height))
        else:
            if aspect_ratio > 1.2:
                new_width = int(target_height * aspect_ratio)
                resized_img = cv2.resize(img, (new_width, target_height))
                start_x = (new_width - target_width) // 2
                resized_img = resized_img[:, start_x:start_x + target_width]
            else:
                new_height = int(target_width / aspect_ratio)
                resized_img = cv2.resize(img, (target_width, new_height))
                start_y = (new_height - target_height) // 2
                resized_img = resized_img[start_y:start_y + target_height, :]

        processed_images.append(resized_img)

    return processed_images

def load_and_process_images(directory, target_width, target_height):
    image_paths = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "gif", "bmp", "tiff")):
                image_paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))

    images = resize_and_pad_images(image_paths, target_width, target_height)
    return images, labels

# Image dataset preparation and splitting
image_dir = "src/excercise05/cats_vs_dogs"
target_width, target_height = 128, 128
images, labels = load_and_process_images(image_dir, target_width, target_height)

# Split dataset into training and test sets
train_test_split = 0.8
split_idx = int(len(images) * train_test_split)

random.seed(42)
combined = list(zip(images, labels))
random.shuffle(combined)
images[:], labels[:] = zip(*combined)

train_images = images[:split_idx]
train_labels = labels[:split_idx]
test_images = images[split_idx:]
test_labels = labels[split_idx:]

label_dict = {"cats": 0, "dogs": 1}
train_labels_encoded = [label_dict[label] for label in train_labels]
test_labels_encoded = [label_dict[label] for label in test_labels]

# Build and train the model
model = Sequential([
    Input(shape=(target_width, target_height, 3)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

num_epochs = 20
batch_size = 32
validation_split = 0.2

best_accuracy = 0.0
best_weights = model.get_weights()

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(num_epochs):
    print(f"Epoche: {epoch + 1}/{num_epochs}")

    combined_train = list(zip(train_images, train_labels_encoded))
    random.shuffle(combined_train)
    train_images[:], train_labels_encoded[:] = zip(*combined_train)

    epoch_loss = 0
    epoch_accuracy = SparseCategoricalAccuracy()

    for step in range(0, len(train_images), batch_size):
        batch_imgs = train_images[step:step + batch_size]
        batch_lbls = train_labels_encoded[step:step + batch_size]

        X_batch = np.array(batch_imgs)
        y_batch = np.array(batch_lbls)

        with tf.GradientTape() as tape:
            preds = model(X_batch, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss += loss.numpy().mean()
        epoch_accuracy.update_state(y_batch, preds)

    history["train_loss"].append(epoch_loss / (len(train_images) // batch_size))
    history["train_accuracy"].append(epoch_accuracy.result().numpy())

    val_loss, val_accuracy = model.evaluate(np.array(test_images), np.array(test_labels_encoded), verbose=0)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_weights = model.get_weights()
        model.save("best_model.h5")

    print(f"Trainingsverlust (Loss): {history['train_loss'][-1]:.4f}, Training-Genauigkeit (Accuracy): {history['train_accuracy'][-1]:.4f}")
    print(f"Validierungsverlust (Loss): {history['val_loss'][-1]:.4f}, Validierungsgenauigkeit (Accuracy): {history['val_accuracy'][-1]:.4f}")

model.load_weights("best_model.h5")

test_loss, test_accuracy = model.evaluate(np.array(test_images), np.array(test_labels_encoded))
print(f"Testgenauigkeit: {test_accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_accuracy"], label="Trainingsgenauigkeit")
plt.plot(history["val_accuracy"], label="Validierungsgenauigkeit")
plt.xlabel("Epochen")
plt.ylabel("Genauigkeit")
plt.title("Genauigkeit des Training und der Validierung")
plt.legend()

plt.tight_layout()
plt.savefig("model_performance.png")
plt.show()