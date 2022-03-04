# to import libraries
import numpy as np
import pandas as pd

# to read data
dataset = pd.read_csv('C:/--your file path here--/mock_fitness_scores_slr.csv')

# to select data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# to print data characteristics
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# to fit simple linear regression using scikit-learn module
from sklearn.linear_model import LinearRegression
LR_model = LinearRegression()
LR_model.fit(x, y)

y_pred = LR_model.predict(x)

print('--------------------------------------------')
from sklearn import metrics
print('LR_model Intercept:', LR_model.intercept_)
print('LR_model Coef:', LR_model.coef_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
print('--------------------------------------------')

# to print summary of results using statsmodels module
from statsmodels import api as sm
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())
print('--------------------------------------------')
print('F-statistic P-value:', est2.f_pvalue)
print('--------------------------------------------')

# to visualize data
from matplotlib import pyplot as plt
plt.scatter(x, y, color='red')
plt.plot(x, LR_model.predict(x), color='blue')
plt.title('training_hours vs vo2_peak')
plt.xlabel('training_hours')
plt.ylabel('vo2_peak')
plt.show()
