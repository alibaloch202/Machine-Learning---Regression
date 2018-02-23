# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:13:41 2017

@author: Ali Baloch
"""
#Polynomial Regression

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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
# poly_reg is going to be the transfroming tool that transform our matrix of feature X
#into the a new metrix of feature that we'r going to call X poly which will be a new matrix of feature
# which will contain only our independent varibale x1 but also x1^2 (x1 square)
poly_reg = PolynomialFeatures(degree = 4) # made a squre with the first column of constan 1
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
y_pred_poly = lin_reg_2.predict(X_poly)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title("Truth Or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'purple')
plt.title("Truth or Bluff (Ploynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# predict a new rsult with Linear Regression
lin_reg.predict(6.5)

#predict a new result with Ploynimial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))