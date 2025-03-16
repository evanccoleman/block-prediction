import numpy as np
import pandas as pd
import h5py
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
    x = handle['matrix_of_200'][:]
print(x.shape)
print(y.shape)
# # Visualizing our single-feature dataset
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, alpha=0.5)
# plt.xlabel('data')
# plt.ylabel('blocks')
# plt.title('blocks vs matrix')
# plt.show()




y = y[0, :].reshape(1, -1)
x = x.reshape(1000, -1)
y = y.T
y = np.ravel(y)
print(x.shape)
print(y.shape)
X_complex, y_complex = x, y
X_train_complex, X_test_complex, y_train_complex, y_test_complex = train_test_split(
    X_complex, y_complex, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_complex)
X_test_scaled = scaler.transform(X_test_complex)


lr_model = LinearRegression()
lr_model.fit(X_train_complex, y_train_complex)

lr_pred = lr_model.predict(X_test_complex)

lr_mse = mean_squared_error(y_test_complex, lr_pred)
lr_r2 = r2_score(y_test_complex, lr_pred)

print(f"Linear Regression Results:")
print(f"Mean Squared Error: {lr_mse}")
print(f"R2 Score: {lr_r2}")
print("Model coefficients:", lr_model.coef_)
print("Model intercept:", lr_model.intercept_)