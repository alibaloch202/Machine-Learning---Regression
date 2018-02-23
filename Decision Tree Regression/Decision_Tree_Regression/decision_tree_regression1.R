#Decision tree Regression 

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


# Fitting dataset into Decision Tree Model
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# predict the new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level= 6.5))


# Visualising the Decision Tree Regression result
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level),0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color ='red') + 
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            color ='blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
  xlab('Levels') +
  ylab('Salary') 
##### shows a problem in graph in a straight line, problem is related to splits, and we have depend variable which 
# is multiples of hundereds and needs the intervals by giving the conditions on splits, for this we can give parameter of condition to model
# rpart.control(minsplit = 1),  after which it also take us to trap, red flag, as it gives more than one splits here and 
# Decision tree splits the independent variables into several intervals, which took the avg of dependent variables but it
# is not splits the intervals properly, so we need to check graphs resolution ####


# Preidict the new result with Regression
y_pred_poly = predict(regressor, data.frame(Level =6.5, 
                                            Level2 =6.5^2, 
                                            Level3 = 6.5^3, 
                                            Level4 = 6.5^4))
