# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding cateorical data
# Encoding the independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Avoiding the dummy vairbale trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # fit multiple linear regressor

#Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination.
import statsmodels.formula.api as sm
# adding the columns of one in the  beginnning as statsmodel library doesn't take account the constant of Multiple linear regression.
X = np.append(arr =np.ones((50, 1)).astype(int), values = X, axis = 1)
# step 2
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # matrix containing all the independent variables, we have to remove the index after each step afterward, thats why we have mentioned every index
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#step 3
regressor_OLS.summary()
# As selected SL = 0.005 andforeach variable/predictor if  P > SL then remove that predictor , otherwise go to finish

# Removing the index 2  as it has higest p value greated then SL= 0.005
X_opt = X[:, [0, 1, 3, 4, 5]] # matrix containing all the independent variables, we have to remove the index after each step afterward, thats why we have mentioned every index
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing the index 1 as it has higest p value greated then SL= 0.005
X_opt = X[:, [0, 3, 4, 5]] # matrix containing all the independent variables, we have to remove the index after each step afterward, thats why we have mentioned every index
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing the index 4 as it has higest p value greated then SL= 0.005
X_opt = X[:, [0, 3, 5]] # matrix containing all the independent variables, we have to remove the index after each step afterward, thats why we have mentioned every index
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing the index 5 as it has higest p value greated then SL = 0.005
X_opt = X[:, [0, 3]] # matrix containing all the independent variables, we have to remove the index after each step afterward, thats why we have mentioned every index
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Afterall the elimination we wont remove independent varibale 









