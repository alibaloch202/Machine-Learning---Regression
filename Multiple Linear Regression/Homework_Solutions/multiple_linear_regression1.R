#Multiple Linear regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fitting Multiple Linear Regression to the Training set
#regressor =lm(formula = Profit~R.D.Spen + Administration + Marketing.Spend + State)
# or  Alternate
regressor =lm(formula = Profit~ .,
              data = training_set)  
  
# Prdiciting the Test set result
y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using Backward Elimination 
regressor =lm(formula = Profit~ R.D.Spend + Administration + Marketing.Spend + State,
              data = dataset)  # R language replace spaces between dataset name/ columns with Dot '.'


regressor =lm(formula = Profit~ R.D.Spend + Administration + Marketing.Spend,
              data = dataset) 
summary(regressor)
y_pred = predict(regressor, newdata = test_set)


regressor =lm(formula = Profit~ R.D.Spend + Marketing.Spend,
              data = dataset) 
summary(regressor)
y_pred = predict(regressor, newdata = test_set)



regressor =lm(formula = Profit~ R.D.Spend,
              data = dataset) 
summary(regressor)
y_pred = predict(regressor, newdata = test_set)
