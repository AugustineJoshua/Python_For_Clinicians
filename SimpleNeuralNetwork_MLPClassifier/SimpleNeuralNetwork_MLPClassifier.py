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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score

X = dataset[['fac_pre', 'fim_pre', 'nihss_pre']]
y = dataset['fim_eff_cat']
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
# no. of hidden layers = start with 2 layers
# no. of hidden neurons = (no. of inputs + outputs)/2
MLP = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=500)
MLP.fit(X_train, y_train)

y_train_pred = MLP.predict(X_train)

print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

# to display loss curve
import matplotlib.pyplot as plt
plt.plot(MLP.loss_curve_)
plt.title("Loss Curve")
plt.xlabel('Iterations or Epochs')
plt.ylabel('Loss or Cost')
plt.show()

# to compute and display ROC curve
fpr, tpr, _ = roc_curve(y_train, y_train_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.02f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.show()

