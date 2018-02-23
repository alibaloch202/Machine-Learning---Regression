# Polynomial Regression
# Data Preprocessing 

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] # select only level and salary column

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
#set.seed(123)
#split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ Level, 
             data = dataset)

#Fitting Ploynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., 
             data = dataset)

# Visualising the Linear Regression results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color ='red') + 
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
             color ='blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Levels') +
  ylab('Salary')

# Visualising the Polynomial Regression result
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color ='red') + 
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color ='blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Levels') +
  ylab('Salary')

# predict the new rsult with Linear Regression
y_pred = predict(lin_reg, data.frame(Level= 6.5))

# Preidict the new result with PPloynomial Regression
y_pred_poly = predict(poly_reg, data.frame(Level =6.5, 
                                           Level2 =6.5^2, 
                                           Level3 = 6.5^3, 
                                           Level4 = 6.5^4))
