# SVR 
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


# Fitting the SVR to dataset
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# predict the new result with Regression
y_pred = predict(regressor, data.frame(Level= 6.5))


# Visualising the  Regression result
library(ggplot2)
#x_grid = seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color ='red') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            color ='blue') +
  ggtitle('Truth or Bluff (SVR Model)') +
  xlab('Levels') +
  ylab('Salary')
# CEO is detect as outlier and SVR didn't consider it as it is very far from other datatypes



# Preidict the new result with Regression
y_pred_poly = predict(regressor, data.frame(Level =6.5, 
                                            Level2 =6.5^2, 
                                            Level3 = 6.5^3, 
                                            Level4 = 6.5^4))
