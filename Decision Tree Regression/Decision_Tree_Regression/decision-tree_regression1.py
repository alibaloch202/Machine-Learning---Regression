# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:53:50 2017

@author: Zeeshan K
"""

# Descion Tree Regression Model


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Just to check weather the employee is telling truth about his/her salary or bluffing
X = dataset.iloc[:, 1:2].values # transforming the column into matrix of feature so that we can apply polynomial rules
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Descion Tree Model to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



#predict a new result with Decision Tree Regression
y_pred = regressor.predict(6.5)

# Visualising the Descion Tree Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, regressor.predict(X), color = 'purple')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the Descion Tree Regression result for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'purple')
plt.title("Truth or Bluff (Ploynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

