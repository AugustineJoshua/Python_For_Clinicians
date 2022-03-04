# to import libraries

import os
import pandas as pd
import numpy as np

# to read data
dataset = pd.read_excel('C:/--your file path here--/mock_database_slgr.xlsx')
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# to fit a simple logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score

X = dataset[['fac_pre']]
y = dataset['fim_eff_cat']
print(X.shape)

LR = LogisticRegression(random_state=0).fit(X, y)

y_pred = LR.predict(X)
cm = confusion_matrix(y_true=y, y_pred=y_pred)
print(cm)
cm_display = ConfusionMatrixDisplay(cm).plot()

# to print model parameters
print('Coefficients:', LR.coef_)
print('Column names:', X.columns)
print(classification_report(y, y_pred))

# to print model summary
import statsmodels.api as sm
X=sm.add_constant(X)
LR=sm.Logit(y, X)
est=LR.fit()
print(est.summary())

# to compute and display ROC curve
fpr, tpr, _ = roc_curve(y, y_pred)
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

