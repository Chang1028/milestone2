setwd('C:/Users/郭钦尧/Desktop/Project 5630')
#install.packages('doParallel')
library(FNN)
library(e1071)
library(caret)
library(doParallel)

data = read.csv("smoking.csv")
data = data[,-c(1,24)]
data$gender = as.numeric(factor(data$gender,levels = c('F','M'), labels = c(0,1)))
data$tartar = as.numeric(factor(data$tartar,levels = c('Y','N'), labels = c(0,1)))
data[,-c(1,24,25)] = apply(data[,-c(1,24,25)], 2, scale)


set.seed(2023)
index = sample(nrow(data),0.8*nrow(data))


misClassError = function(k.input){
  knn.img <- knn(train = data[index, -25],
                 test = data[-index, -25], 
                 cl = data[index, 25], 
                 k = k.input)
  return(mean(data[-index, 25] != (as.numeric(knn.img))-1))
}

misClassError.all = apply(as.matrix(c(1:20)*10),1,misClassError)

plot(c(1:20)*10, misClassError.all,type = 'l')


# svm part
data <- read.csv("smoking.csv")
data <- data[,-c(1,24)]
data$gender <- as.numeric(factor(data$gender,levels = c('F','M'), labels = c(0,1)))-1
data$tartar <- as.numeric(factor(data$tartar,levels = c('Y','N'), labels = c(0,1)))-1
data$smoking = factor(data$smoking)
set.seed(2023)
index = sample(nrow(data),0.8*nrow(data))
data.train = data[index,]
data.test = data[-index,]

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

# decision tree
data <- read.csv("smoking.csv")
data <- data[,-c(1,24)]
data$gender <- factor(data$gender,levels = c('F','M'), labels = c(0,1))
data$tartar <- factor(data$tartar,levels = c('Y','N'), labels = c(0,1))
data$smoking = factor(data$smoking)
set.seed(2023)
index = sample(nrow(data),0.8*nrow(data))
data.train = data[index,]
data.test = data[-index,]

library(rpart)
library(caret)
library(rpart.plot)
library(pROC)
folds = createFolds(1:nrow(data),k = 10)
c = seq(0.001,0.1,0.002)
cr = matrix(rep(0,500),nrow = 10)
for (i in 1:10) {
  for (j in 1:length(c)) {
    fit.tree.2 = rpart(smoking ~ ., data = data.train[unlist(folds[i]),], control = rpart.control(cp = c[j]))
    pre.tree.2 = predict(fit.tree.2, newdata = data.train[-unlist(folds[i]),], type = 'class')
    cr[i,j] = mean(pre.tree.2 == data.train[-unlist(folds[i]),]$smoking)
  }
}
c.best = c[which.max(apply(cr, 2, mean))]
fit.tree = rpart(smoking ~ ., data = data.train, control = rpart.control(cp = c.best))

rpart.plot(fit.tree,cex = 0.8, fallen.leaves = F)

pre.tree = predict(fit.tree, newdata = data.test, type = 'class')
pre.tree.num = predict(fit.tree, newdata = data.test, type = 'prob')
mean(pre.tree == data.test$smoking)
r = roc(data.test$smoking,pre.tree.num[,1])
auc(r)
plot(r)
# random forest
library(randomForest)
fit.rf <- randomForest(smoking~., data = data.train)
pre.rf <- predict(fit.rf, newdata = data.test, type = 'class')
mean(pre.rf == data.test$smoking)


pre.rf.numeric = predict(fit.rf, newdata = data.test, type = 'prob')
roc(data.test$smoking, pre.rf.numeric[,1])
varImpPlot(fit.rf)
hist(fit.rf$oob.times)
plot(fit.rf)

# linear regression
data <- read.csv("smoking.csv")
data <- data[,-c(1,2,26,24,27)]
data = apply(data,2,scale)
data = as.data.frame(data)

set.seed(2023)
index = sample(nrow(data),0.8*nrow(data))
data.train = data[index,]
data.test = data[-index,]
fit = lm(age ~ .,data = data.train)
summary(fit)

data = data[,-c(17,15,13)]
data.train = data[index,]
data.test = data[-index,]
fit = lm(age ~0+.,data = data.train)
summary(fit)
pre = predict(fit, newdata = data.test)
1-sum((pre-data.test$age)^2)/var(data.test$age)/(length(data.test)-1)
plot(data.test$waist.cm.
     ,data.test$age)
