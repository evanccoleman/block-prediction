import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

#generate random seed
np.random.seed(42)

with h5py.File('./synthetic_data.h5', 'r') as handle:
    #labels = np.array(handle['labels_for_10'])
    #value = labels[0, 0]
    #dim = labels.shape[0]
    #labels = np.full((1, dim), value, dtype=int)
    y = handle['labels_for_200'][:]
    print(y.shape)
    y = y[0, :].reshape(1, -1)
    print(y.shape)
    x = handle['matrix_of_200'][:]
    print(x.shape)
x = x.reshape(1000, -1)
y = y.reshape(1000, 1) # hiii

print(y.shape)
print(x.shape)

X_complex, y_complex = x, y
y_complex = np.ravel(y_complex)
X_train_complex, X_test_complex, y_train_complex, y_test_complex = train_test_split(
    X_complex, y_complex, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_complex)
X_test_scaled = scaler.transform(X_test_complex)



simple_nn = MLPRegressor(
    hidden_layer_sizes=(),     # No hidden layers - same as linear regression
    activation='identity',     # Linear activation
    solver='adam',            # Adam optimizer often works better than default
    learning_rate_init=0.001, # Smaller learning rate for stability
    max_iter=2000,           # More iterations to ensure convergence
    tol=1e-8,               # Stricter convergence criteria
    batch_size='auto',      # Let it optimize batch size
    early_stopping=True,    # Enable early stopping
    validation_fraction=0.1, # Use some data for validation
    n_iter_no_change=50,    # Be patient with convergence
    random_state=42
)

simple_nn.fit(X_train_scaled, y_train_complex)
simple_nn_pred = simple_nn.predict(X_test_scaled)

simple_nn_mse = mean_squared_error(y_test_complex, simple_nn_pred)
simple_nn_r2 = r2_score(y_test_complex, simple_nn_pred)



print("Neural Network (No Hidden Layers) Results:")
print(f"MSE: {simple_nn_mse:.2f}")
print(f"R² Score: {simple_nn_r2:.3f}")


mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # Two hidden layers, one with 64 neurons and one with 32 neurons
    activation='relu',  # ReLU activation allows for non-linear patterns
    max_iter=10000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train_complex)
mlp_pred = mlp.predict(X_test_scaled)

mlp_mse = mean_squared_error(y_test_complex, mlp_pred)
mlp_r2 = r2_score(y_test_complex, mlp_pred)

print("\nNeural Network (With Hidden Layers) Results:")
print(f"MSE: {mlp_mse:.2f}")
print(f"R² Score: {mlp_r2:.3f}")

deep_mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='logistic',
    learning_rate_init=0.01,
    max_iter=10000,
    random_state=42
)

deep_mlp.fit(X_train_scaled, y_train_complex)
deep_mlp_pred = deep_mlp.predict(X_test_scaled)

deep_mlp_mse = mean_squared_error(y_test_complex, deep_mlp_pred)
deep_mlp_r2 = r2_score(y_test_complex, deep_mlp_pred)

print("\nDeep Neural Network (With Hidden Layers) Results:")
print(f"MSE: {deep_mlp_mse:.2f}")
print(f"R² Score: {deep_mlp_r2:.3f}")


tf_seq = keras.Sequential([
    keras.Input(shape=(X_complex.shape[1],)),
    layers.Dense(128, activation='selu'),  # First hidden layer
    layers.Dense(64, activation='selu'),
    layers.Dense(32, activation='selu'),
    layers.Dense(16, activation='selu'),
    layers.Dense(1, activation='linear')  # Output layer (for classification)
])

tf_seq.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

X_complex_seq, y_complex_seq = x, y
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_complex_seq, y_complex_seq, test_size=0.2, random_state=42
)

history = tf_seq.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)
predictions = tf_seq.predict(X_test_seq)

print(predictions)

test_loss, test_mae, test_mse = tf_seq.evaluate(X_test_seq, y_test_seq)
print(f"Test MAE FOR SEQ: {test_mae}")
print(f"Test MSE SEQ: {test_mse}")


print("\n")
tf_seq.summary()

