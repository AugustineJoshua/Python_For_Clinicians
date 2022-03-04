# to import libraries

import os
import pandas as pd
import numpy as np

# to read data
dataset = pd.read_excel('C:/--your file path here--/mock_database_mlgr.xlsx')
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# to fit binomial logistic regression - split into training and testing datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score

X = dataset[['fac_pre', 'fim_pre', 'nihss_pre']]
y = dataset['fim_eff_cat']
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

LR = LogisticRegression(random_state=0).fit(X_train, y_train)

y_train_pred = LR.predict(X_train)
cm = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
print(cm)
cm_display = ConfusionMatrixDisplay(cm).plot()

# to print model parameters
print('Coefficients:', LR.coef_)
print('Column names:', X.columns)
print(classification_report(y_train, y_train_pred))

# to print model summary
import statsmodels.api as sm
X_train=sm.add_constant(X_train)
LR=sm.Logit(y_train, X_train)
est=LR.fit()
print(est.summary())

# to compute and display ROC curve
fpr, tpr, _ = roc_curve(y_train, y_train_pred)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.02f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.show()

