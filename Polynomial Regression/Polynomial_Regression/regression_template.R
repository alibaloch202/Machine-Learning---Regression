# Regression Template

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


# Create your Regressor Here



# Visualising the  Regression result
# Visualising the  Regression result
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color ='red') + 
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = Level = x_grid)),
            color ='blue') +
  ggtitle('Truth or Bluff (Random Forest Regression Model)') +
  xlab('Levels') +
  ylab('Salary')

# predict the new result with Regression
y_pred = predict(lin_reg, data.frame(Level= 6.5))

# Preidict the new result with Regression
y_pred_poly = predict(regressor, data.frame(Level =6.5, 
                                           Level2 =6.5^2, 
                                           Level3 = 6.5^3, 
                                           Level4 = 6.5^4))
