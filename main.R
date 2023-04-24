# Packages for tree methods
library(rpart)
library(rpart.plot)
library(caret)        # an aggregator package for performing many machine learning models
library(ggplot2)
library(pROC)

# Packages for Random Forest
library(randomForest)

# Packages for gbm
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(lime)         # model visualization

# Packages for KNN
library(FNN)
library(e1071)
library(caret)

##----------- Input data--------------
data <- read.csv("smoking.csv")
# Delete "ID" and "oral" columns
data <- data[,-c(1,24)]
# Convert "gender" and "tartar" to 0-1 dummy variables
data$gender = as.numeric(factor(data$gender,levels = c('F','M'), labels = c(0,1)))
data$tartar = as.numeric(factor(data$tartar,levels = c('Y','N'), labels = c(0,1)))
data$smoking = factor(data$smoking)
# scale the other variables
data[,-c(1,24,25)] = apply(data[,-c(1,24,25)], 2, scale)
# Split the data into training and test sets
m <- nrow(data)
set.seed(2023)
index <- sample(m, floor(0.8*m))
data.train <- data[index,]
data.test <- data[-index,]


## ------------KNN----------------------
misClassError = function(k.input){
  knn.fit <- knn(train = data[index, -25],
                 test = data[-index, -25], 
                 cl = data[index, 25], 
                 k = k.input)
  return(mean(data[-index, 25] != (as.numeric(knn.img))-1))
}
# The prediction error rate on the whole dataset 
misClassError.whole = apply(as.matrix(c(1:20)*10),1,misClassError)
# Plot the curve with x-axis being "k", y-axis being error rate
plot(c(1:20)*10, misClassError.all,type = 'l')

## ------------svm----------------------
# Due to computational cost, we only select 10000 data to perform svm
index.tune = sample(nrow(data), 10000)
data.tune = data[index.tune,]
svmTune <- train(
  y = data.tune$smoking,
  x = data.tune[, -25],
  method = "svmLinear2",
  preProcess = c("center", "scale"),
  tuneGrid = data.frame(cost = seq(0.01,10,length.out = 20)),
  trControl = trainControl(
    method = "repeatedcv",
    repeats = 1, number = 10
  )
)
svmTune$bestTune

cr = c()
c = seq(0.01,1,length.out = 10)
for(i in 1:10){
  svm.fit = svm(smoking~.,data = data.train, kernel = 'linear',cost = c[i])
  svm.pred = predict(svm.fit, data[-index,])
  cr = c(cr,mean(svm.pred == data[-index,25]))
}
svm.fit = svm(smoking~.,data = data.train, kernel = 'linear',cost = 9)
print(svm.fit)
svm.pred = predict(svm.fit, data[-index,])
cr = mean(svm.pred == data[-index,25])
1 - cr


## ------------Decision Tree------------
# Create cross-validation folds
folds = createFolds(1:nrow(data),k = 10)
# c is the tuning parameter in the tree model
c = seq(0.001,0.1,0.002)
# cr variable stores the error rate results
cr = matrix(rep(0,10*length(c)),nrow = 10)
# Run the 10-fold cross-validation for 50 parameters
for (i in 1:10) {
  for (j in 1:length(c)) {
    fit.tree = rpart(smoking ~ ., data = data.train[unlist(folds[i]),], control = rpart.control(cp = c[j]))
    pred.tree = predict(fit.tree, newdata = data.train[-unlist(folds[i]),], type = 'class')
    cr[i,j] = mean(pred.tree == data.train[-unlist(folds[i]),]$smoking)
  }
}
# Choose the best c with the max accuracy
c.best = c[which.max(apply(cr, 2, mean))]
# Fit the tree model with the chosen c
fit.tree.best = rpart(smoking ~ ., data = data.train, control = rpart.control(cp = c.best))
# Make predictions on the test data
pred.tree = predict(fit.tree.best, newdata = data.test, type = 'class')
# Plot the tree
rpart.plot(fit.tree.best,cex = 0.8, fallen.leaves = F)
# Draw the ROC curve and calculate AUC
pred.tree.prob = predict(fit.tree.best, newdata = data.test, type = 'prob')
mean(pred.tree == data.test$smoking)
r = roc(data.test$smoking, pre.tree.prob[,1])
auc(r)
plot(r)


### -----------Random Forest------------
# fit the Random Forest model
fit.rf <- randomForest(smoking~., data = data.train)
pred.rf <- predict(fit.rf, newdata = data.test, type = 'class')
# Prediction accuracy on test set
mean(pred.rf == data.test$smoking)
# Draw the ROC curve and calculate AUC
pred.rf.prob = predict(fit.rf, newdata = data.test, type = 'prob')
roc(data.test$smoking, pre.rf.numeric[,1])
# Draw the variable importance plot
varImpPlot(fit.rf)
hist(fit.rf$oob.times)
plot(fit.rf)


## -----------Boosting------------------


## ----------Linear Regression------------
data <- read.csv("smoking.csv")
# Delete "ID" and "oral" columns
data <- data[,-c(1,24)]
# Convert "gender" and "tartar" to 0-1 dummy variables
data$gender = as.numeric(factor(data$gender,levels = c('F','M'), labels = c(0,1)))
data$tartar = as.numeric(factor(data$tartar,levels = c('Y','N'), labels = c(0,1)))
data$smoking = factor(data$smoking)
colnames(data)[c(3,4)] <- c("height", "weight")
# scale the other variables
#data[,-c(1,24,25)] = apply(data[,-c(1,24,25)], 2, scale)
# Split the data into training and test sets
m <- nrow(data)
set.seed(2023)
index <- sample(m, floor(0.8*m))
data.train <- data[index,]
data.test <- data[-index,]

index.lm <- sample(m, 10000)
data.lm <- data[index.lm, ]
lm.fit <- lm(HDL~weight, data = data.lm)
summary(lm.fit)
pred.lm <- predict(lm.fit, data.lm)
df <- data.frame(x = data.lm$weight, y1 = data.lm$HDL, y2 = pred.lm)
ggplot()+
  geom_point(data = df, mapping = aes(x = x, y = y1))+
  geom_point(data = df, mapping = aes(x = x, y = y2), color = "red", size = 1)


## ----------Kernel Smoothing-------------
library(splines2)
library(tidyr)
fit.spline <- lm(HDL ~ 0 + bSpline(weight, knots = quantile(weight, c(0.25, 0.5, 0.75)), degree = 3, intercept = T), data = data.lm)
x <- seq(30, 140, by = 1)
x <- data.frame(weight = x)
y.spline <- predict(fit.spline, x)
spline.curve <- data.frame(x, y.spline)
ggplot()+
  geom_point(data = data.lm, mapping = aes(x = weight, y = HDL))+
  geom_point(data = spline.curve, mapping = aes(x = weight, y = y.spline), color = "red", size = 0.5)

## ----------Group Mean----------------
data.mean <- aggregate(data$HDL, by=list(weight=data$weight), FUN=median)
fit.lm <- lm(x~weight, data = data.mean)
pred.lm <- predict(fit.lm, data.mean)
data.mean <- cbind(data.mean, pred.lm)
ggplot(data = data.mean)+
  geom_point(mapping = aes(x = weight, y = x))+
  geom_point(mapping = aes(x = weight, y = pred.lm), color = "red", size = 1)


library(splines2)
library(tidyr)
fit.spline <- lm(HDL ~ 0 + bSpline(weight, knots = quantile(weight, c(0.25, 0.5, 0.75)), degree = 3, intercept = T), data = data.lm)
x <- seq(30, 140, by = 1)
x <- data.frame(weight = x)
y.spline <- predict(fit.spline, x)
spline.curve <- data.frame(x, y.spline)
ggplot()+
  geom_point(data = data.lm, mapping = aes(x = weight, y = HDL))+
  geom_point(data = spline.curve, mapping = aes(x = weight, y = y.spline), color = "red", size = 0.5)







