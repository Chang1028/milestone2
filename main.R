# Packages for tree methods
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization

# Packages for 

##----------- Input data--------------
data <- read.csv("smoking.csv")
data <- data[,-c(1,24)]
data$gender <- as.factor(data$gender)
data$tartar <- as.factor(data$tartar)
data$smoking <- as.factor(data$smoking)
data[,-c(1,24,25)] <- apply(data[,-c(1,24,25)], 2, scale)

# treat the categorical variables and outcome variable as.factor
data$gender <- as.factor(data$gender)
data$tartar <- as.factor(data$tartar)
data$smoking <- as.factor(data$smoking)

# treat the categorical variables as dummy variables
dummies <- dummyVars(~ ., data=data[,-25])
c2 <- predict(dummies, data[,-25])
data.new <- as.data.frame(cbind(c2, data$smoking))
data.new <- data.new[, -c(1,25)]
colnames(data.new)[1] <- "gender"
colnames(data.new)[24] <- "tartar"
colnames(data.new)[25] <- "smoking"

m <- nrow(data)
set.seed(2023)
index <- sample(1:m, floor(0.8*m))
training.data <- data[index,]
training.data.dummy <- data.new[index,]
test.data <- data[-index,]
test.data.dummy <- data.new[-index,]


## ------------Decision Tree------------
tree.out <- tree(smoking~., data = training.data)
plot(tree.out)
text(tree.out, pretty = 0)
pred.tree <- predict(tree.out, test.data, type = "class")
table(test.data$smoking, pred.tree)
(5669+2387)/11139
### -----------Cross Validation for Decision Tree---------
cv_carseats <- cv.tree(tree.out, K = 10, FUN = prune.misclass)
plot(cv_carseats$size, cv_carseats$dev, type = "b")
prune_carseats = prune.misclass(tree.out, best = cv_carseats$size[which.min(cv_carseats$dev)])
plot(prune_carseats)
text(prune_carseats, pretty = 0)
