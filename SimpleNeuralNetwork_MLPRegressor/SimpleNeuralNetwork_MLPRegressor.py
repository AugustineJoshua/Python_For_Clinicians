# to import libraries

import os
import pandas as pd
import numpy as np

# to read data
dataset = pd.read_excel('C:/--your file path here--/mock_database_snn_mlp.xlsx')
# print(dataset.shape)
# print(dataset.describe())
# print(dataset.info())
# print(dataset.head())

# to fit a simple neural network

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

X = dataset[['fac_pre', 'fim_pre', 'nihss_pre']]
y = dataset['fim_eff']
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.neural_network import MLPRegressor
# no. of hidden layers = start with 2 layers
# no. of hidden neurons = (no. of inputs + outputs)/2
MLP = MLPRegressor(hidden_layer_sizes=(5, 5, 5), max_iter=2000)
MLP.fit(X_train_scaled, y_train)

# to evaluate model

y_pred = MLP.predict(X_test_scaled)

df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_temp.head())

import matplotlib.pyplot as plt
df_temp = df_temp.head(25)
df_temp.plot(kind='bar', figsize=(7, 5))
plt.grid(which='major', linestyle='-', linewidth='0.2', color='orange')
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='green')
plt.show()

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

# to display loss curve
plt.plot(MLP.loss_curve_)
plt.title("Loss Curve")
plt.xlabel('Iterations or Epochs')
plt.ylabel('Loss or Cost')
plt.show()

