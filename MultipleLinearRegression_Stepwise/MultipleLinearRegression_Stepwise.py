# to import libraries

import os
import pandas as pd
import numpy as np

# to read data
dataset = pd.read_excel('C:/--your file path here--/mock_database_mlr.xlsx')
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# to fit multiple linear regression - stepwise (manual), add/remove variables based on p-value <0.05

import statsmodels.api as sm
import statsmodels.tools

X = dataset[['age', 'moca_pre', 'nihss_pre']]
y = dataset['fim_eff']
X = statsmodels.tools.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

# to visualize data
import seaborn as sns
sns.pairplot(X)
import matplotlib.pyplot as plt
plt.show()

# to fit multiple linear regression - stepwise (semi-automated), based on p-value <0.05

import statsmodels.api as sm

# getting column names
X = ['age', 'moca_pre', 'nihss_pre']
y = dataset['fim_eff']

# creating function to get model statistics
def get_stats():
    x1 = dataset[X]
    x2 = sm.add_constant(x1)
    model = sm.OLS(y, x2).fit()
    print(model.summary())
get_stats()

X.remove('moca_pre')
get_stats()

X.remove('age')
get_stats()

# to fit multiple linear regression - stepwise (automated)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, max_error

X = dataset[['age', 'moca_pre', 'nihss_pre']]
y = dataset['fim_eff']
print(X.shape)

from sklearn.feature_selection import SequentialFeatureSelector as SFS
sfs_forward = SFS(estimator=LinearRegression(), direction='backward', scoring='r2', cv=None).fit(X, y)
print(sfs_forward.transform(X).shape[:])
print(sfs_forward.get_feature_names_out(input_features=None))
print(sfs_forward.get_support())
print(sfs_forward.get_params(deep=True))

# to visualize data
import seaborn as sns
sns.regplot(x='moca_pre', y='fim_eff', data=dataset, fit_reg=True)
sns.lmplot(x='moca_pre', y='fim_eff', data=dataset, fit_reg=True, x_jitter=0.05)
sns.lmplot(x='moca_pre', y='fim_eff', data=dataset, fit_reg=True, x_jitter=None, x_estimator=np.mean)
sns.residplot(x='moca_pre', y='fim_eff', data=dataset)
import matplotlib.pyplot as plt
plt.show()

