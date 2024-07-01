import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# 1. Erstellung eines Klassifikationsdatensatzes

# Zwei-Klassen-Datensatz generieren
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=1.5, random_state=42)

# Visualisierung des Datensatzes
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Klassifikationsdatensatz')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Daten als DataFrame speichern und als CSV exportieren
df = pd.DataFrame(data=X, columns=['Feature1', 'Feature2'])
df['Label'] = y
csv_path = 'classification_data.csv'
df.to_csv(csv_path, index=False)

# 2. Neuronales Netzwerk mit TensorFlow erstellen


# Modell definieren
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Modell als Bild speichern
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# 3. Benutzerdefinierte Trainingsroutine implementieren

# Trainingsparameter
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
batch_size = 32
epochs = 10
accuracy_metric = tf.keras.metrics.BinaryAccuracy()

# Trainingsdaten vorbereiten
dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Trainingsroutine
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss_avg = tf.keras.metrics.Mean()

    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch_train, training=True)
            loss = loss_fn(y_batch_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update training metric.
        epoch_loss_avg.update_state(loss)
        accuracy_metric.update_state(y_batch_train, predictions)

    # End epoch
    loss_history.append(epoch_loss_avg.result())
    accuracy_history.append(accuracy_metric.result())
    print("Epoch %d: Mean loss: %.4f - Accuracy: %.4f" % (epoch, epoch_loss_avg.result(), accuracy_metric.result()))

    # Reset training metrics at the end of each epoch
    accuracy_metric.reset_states()

# Plotting der Ergebnisse
plt.plot(loss_history)
plt.title('Trainingsverlust')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_history.png')
plt.show()

plt.plot(accuracy_history)
plt.title('Trainingsgenauigkeit')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy_history.png')
plt.show()


# 4. Bestes Modell speichern

# Überprüfen, ob das Modell verbessert wurde
best_accuracy = 0.0
for epoch_acc in accuracy_history:
    if epoch_acc > best_accuracy:
        best_accuracy = epoch_acc
        # Modell speichern
        model.save('best_model.h5')
        print("Model improved and saved at epoch: ", accuracy_history.index(epoch_acc), "with accuracy:", epoch_acc)
