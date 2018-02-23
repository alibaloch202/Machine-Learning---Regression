# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:42:25 2017

@author: Ali Baloch
"""

# Support Vector Regression SVR

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))  


# Fitting Regression Model to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predict a new result with Ploynimial Regression
# inversing the tranformation to get the real result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) # array of one line and one column: due to double brackets
# inversing the tranformation to get the real result

# Visualising the SVR results

plt.scatter(X, y, color = 'black')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# outlier detected for CEO


#Visualising the SVR result for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'purple')
plt.title("Truth or Bluff (SVR)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


