library(caret)
library(randomForest)
library(cluster)
library(gplots)
library(RColorBrewer)
library(Metrics)
library(pROC)
url = 'https://raw.githubusercontent.com/guru99-edu/R-Programming/master/titanic_data.csv'
titanic_data = read.csv(url, header = TRUE, sep = ",")

### data inspection ###
sum(titanic_data$pclass=="?")
head(titanic_data, 10)

### data cleaning ###
titanic_data = titanic_data[,-1] # exlcude 1st column

titanic_data[titanic_data == "?"] = NA # replace ? with NA's

head(titanic_data, 10)

str(titanic_data)

titanic_data$relatives = titanic_data$parch + titanic_data$sibsp # create a new variable relative that combines siblings and parents

titanic_data$survived = ifelse(titanic_data$survived == 1, "Survived", "Passed")
titanic_data$pclass = as.factor(titanic_data$pclass)
titanic_data$survived = as.factor(titanic_data$survived)
titanic_data$sex = as.factor(titanic_data$sex)
titanic_data$embarked = as.factor(titanic_data$embarked)
titanic_data$cabin = as.factor(titanic_data$cabin)
titanic_data$age = as.numeric(titanic_data$age)
titanic_data$fare = as.numeric(titanic_data$fare)

titanic_final = titanic_data[,-c(3, 6, 7, 8, 10, 12)] # exclude variables not useful for the analysis
str(titanic_final)

### impute values for the NAs
### in theory 4-6 iterations is enough
### OOB error rate should get smaller if the estimates are improving
set.seed(1)

titanic_imputed = rfImpute(survived ~. , data = titanic_final, iter = 6)
sum(is.na(titanic_imputed))
View(titanic_imputed)

### split the data ###

set.seed(1)
train.prop = 0.75
indice = sample (1: nrow(titanic_imputed), nrow(titanic_imputed)*train.prop)
train = titanic_imputed[indice ,]
test= titanic_imputed[- indice ,]


### Bagging ###

## special case of random forest with m = p
## mtry = 8, means that all 8 predictors should be considered for each split of the tree 
set.seed(1)

bag.titanic = randomForest(survived ~. , data = train,
                           mtry = 7, importance = TRUE)
bag.titanic
# how well the bagged model perform on the test set?
bag.pred = predict(bag.titanic, newdata = test, type = "class")
bag.pred

# test the performance of the bagging
bag.pred = predict(bag.titanic, newdata = test, type = "prob")
roc(test$survived, bag.pred, plot = TRUE)

auc(test$survived, bag.pred[,2])
ce(test$survived, bag.pred)

# test the accuracy of the model
table(test$survived, bag.pred)
### Random Forest ###

### we want to return the proximity matrix so we set proximity = TRUE, to use this to cluster
### the samples
### type denotes that the model is used for classification
### 500 trees are in the random forest
### classification trees by default have a setting of square root of the number of variables (8)
### 1- 20.18% (OOB error rate) of the OOB samples were correctly classified
set.seed(1)
rf.titanic = randomForest(survived ~ ., data = train, proximity =  TRUE)
rf.titanic


### random to some extent ###
### select variables and subsets of data probabilistacally ###
### ntree argument specifies the number of trees to create in the forest ###
### rule of thumb, is to set ntree with a number so as to have 5-10 trees per obs for small data sets ###
### RF fits many trees, where each tree is optimized for a portion of the data ###
### uses the remainder of the data (out of bag OOB) to assess the tree's performance ###
### class error is an indicator of the error rate in the OOB data
### based on that we may see similar patterns in our test data ###
### when an obs is classified, it is assigned to the group that is predicted by the greatest
### number of trees within the ensemble
### testing whether adding more trees will decrease error rate
### the graph stabilize after 500 trees so no point in increasing the number of trees
set.seed(1)
rf.titanic = randomForest(survived ~ ., data=train, ntree=4905, proximity=TRUE)
rf.titanic


oob.error.data = data.frame(
  Trees=rep(1:nrow(rf.titanic$err.rate), times=3),
  Type=rep(c("OOB", "Survived", "Passed"), each=nrow(rf.titanic$err.rate)),
  Error=c(rf.titanic$err.rate[,"OOB"], 
          rf.titanic$err.rate[,"Survived"], 
          rf.titanic$err.rate[,"Passed"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

### we choose ntree = 1500 based on the OOB graphs since after that point the OOB estimate of error does change that much

### check whether the default number of variables at each internal node in the tree is optimal?
### we would re run the random forest with ntree = 1500 and mtry = 3
oob.values = vector(length=6)
for(i in 1:6) {
  temp.model = randomForest(survived ~ ., data=train, mtry=i, ntree=1500)
  oob.values[i] = temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

### final random forest model ###
set.seed(1)
rf.titanic = randomForest(survived ~ ., data=train, ntree=1500, mtry = 3, proximity=TRUE)
rf.titanic

### Prediction ###
######################################################################################
######################################################################################
######################################################################################
rf.titanic.pred = predict(rf.titanic, test, type = "class")
rf.titanic.pred
clusplot(test[, -1], rf.titanic.pred, color = TRUE, shade = TRUE,
         labels = 4, lines = 0, main = "Random Forest classification, holdout data")

### overall prediction performance
mean(test$survived == rf.titanic.pred)

### confusion matrix 
table(test$survived, rf.titanic.pred)


#################################################################################
#################################################################################
#################################################################################

### ploting random forest
### in general the error rates decrease when we have more trees


### creating an mds plot
distance.matrix =as.dist(1-rf.titanic$proximity)
mds.stuff = cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)

mds.var.per = round(mds.stuff$eig/sum(mds.stuff$eig)*100,1)

mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=train$survived)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")

################################### Variable Importance ###############################################
#######################################################################################################
#######################################################################################################

set.seed(1)
rf.titanic.imp = randomForest(survived ~. , data = train, ntree = 1500, mtry = 3, importance = TRUE)
### variable importance by segment ###
### age is important for all segments, while gender not ###
### meandecreaseaccuracy = permutation measure of impact on accuracy
### if a variable is important then its performance will degrade when its observed values are randomly permuted ###
### meandecreasegini = assessment of the variable's ability to assist classification better than chance labeling ###
importance(rf.titanic.imp)

### graphical representation of the importance 
varImpPlot(rf.titanic.imp, main = "Variable Importance by segment")

### plot the importance for variables by segment ###
### the first 2 columns[,1:2] represent the variable - by - segment ###
### highlights the importance of age and kids to predict Travelers

heatmap.2(t(importance(rf.titanic.imp)[, 1:2]),
          col = brewer.pal(9, "Blues"),
          dend = "none", trace = "none", key = FALSE,
          margins = c(10, 10),
          main = "Variable Importance by segment")
############################################################################################
############################################################################################
############################################################################################

### Inspect the distribution of predictions for individual cases ###
### predict.all argument to get the estimate of every tree for every case in the test data ###
### saved in the $individual element of the result object
### each row collects the predictions for one case across all trees (on the columns) ###
rf.titanic.pred.all = predict(rf.titanic, test, predict.all = TRUE)


### the predictions of the first 5 cases in the test set 
### highest probability indicates where each case is assigned in each class
apply(rf.titanic.pred.all$individual[1:5,], 1, function(x) prop.table(table(x)))


### proposed and actual segments 
seg.summ = function(data , groups) {
  aggregate(data , list(groups), function(x) mean(as.numeric(x)))
}
# proposed segments
seg.summ(test, rf.titanic.pred)
# actual segments
seg.summ(test, test$survived)