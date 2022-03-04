# to import libraries

import os
import pandas as pd
import numpy as np

# to read data
dataset = pd.read_excel('C:/--your file name here--/mock_database_slr.xlsx')
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# to fit simple linear regression - option 1

# to select data
X = dataset.loc[:, ['vo2_peak']] # [all_rows:, selected column]
y = dataset.loc[:, ['training_hours']] # [all_rows:, selected column]

from statsmodels import api as sm
x2 = sm.add_constant(X)
model = sm.OLS(y, x2)
est = model.fit()
print(est.summary())

# to visualize data
import seaborn as sns
sns.regplot(x=X, y=y, data=dataset)
import matplotlib.pyplot as plt
plt.show()

# to fit simple linear regression -option 2

# to select data
X = dataset.loc[:, ['vo2_peak']] # [all_rows:, selected column]
y = dataset.loc[:, ['training_hours']] # [all_rows:, selected column]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

print('Model Intercept:', model.intercept_)
print('Model Coef:', model.coef_)
print('R_square:', model.score(X, y))

y_pred = model.predict(X)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

from sklearn.feature_selection import f_regression
freg = f_regression(X, y.values.ravel()) # using ravel function here to obtain a flattened array
p = freg[1]
print('F-statistic P-value:', p.round(4))

# to visualize data
import seaborn as sns
sns.regplot(x=X, y=y, data=dataset)
import matplotlib.pyplot as plt
plt.show()

