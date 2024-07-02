import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Daten generieren
np.random.seed(42)
X = 2 * np.pi * np.random.rand(1000, 1)  # Zufällige Datenpunkte im Bereich 0 bis 2pi
y = np.sin(X).ravel() + 0.1 * np.random.randn(1000)  # Sinusfunktion mit Rauschen

# Daten als DataFrame speichern und als CSV exportieren
df = pd.DataFrame(data={"Feature": X.flatten(), "Target": y})
csv_path = 'regression_data.csv'
df.to_csv(csv_path, index=False)

# Daten visualisieren
plt.scatter(X, y)
plt.title('Generierter Regressionsdatensatz')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.show()

# Modell definieren
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Modell als Bild speichern
tf.keras.utils.plot_model(model, to_file='regression_model.png', show_shapes=True, show_layer_names=True)
# Trainingsparameter
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
batch_size = 32
epochs = 100
train_dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Trainingsroutine
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch_train, training=True)
            loss = loss_fn(y_batch_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("Training loss at step %d: %.4f" % (step, float(loss)))
# Modell verwenden, um Vorhersagen zu machen
X_test = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y_pred = model.predict(X_test)

# Vorhersagen plotten
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='Originaldaten')
plt.plot(X_test, y_pred, color='red', label='Modellvorhersage')
plt.title('Regressionsanalyse mit neuronalen Netzwerken')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.savefig('learned_regression_curve.png')
plt.show()
