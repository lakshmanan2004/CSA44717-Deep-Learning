import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

# Generate some dummy data
np.random.seed(0)
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=(1000, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the neural network model with Tanh activation
model_tanh = keras.Sequential([
    keras.layers.Dense(64, activation='tanh', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_tanh.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_tanh.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Define the neural network model with Sigmoid activation
model_sigmoid = keras.Sequential([
    keras.layers.Dense(64, activation='sigmoid', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_sigmoid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_sigmoid.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Define the neural network model with Linear activation
model_linear = keras.Sequential([
    keras.layers.Dense(64, activation='linear', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_linear.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_linear.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Define the neural network model with ReLU activation
model_relu = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_relu.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_relu.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
