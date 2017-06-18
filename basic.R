# Let's try to recreate useful stuff for Kaggle Mercedes challenge.

library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('grid') # visualisation
library('gridExtra') # visualisation
library('corrplot') # visualisation
library('ggfortify') # visualisation
library('dplyr') # data manipulation
library('readr') # data input
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('xgboost') # classifier

setwd("~/Mercedes")
train <- read_csv('train.csv')
test  <- read_csv('test.csv')

train <- train %>%
  mutate_each(funs(factor), c(X0,X1,X2,X3,X4,X5,X6,X8))
test <- test %>%
  mutate_each(funs(factor), c(X0,X1,X2,X3,X4,X5,X6,X8))


train2=train
nr=nrow(train2)
train2$y=NULL
# train2$diff=NULL
# train2$ygroup=NULL
# train2$y82=NULL
test2=test
full=rbind(train2,test2)
full$X0<-as.integer(full$X0)
full$X1<-as.integer(full$X1)
full$X2<-as.integer(full$X2)
full$X3<-as.integer(full$X4)
full$X4<-as.integer(full$X5)
full$X5<-as.integer(full$X5)
full$X6<-as.integer(full$X6)
full$X8<-as.integer(full$X8)
full2=as.matrix(full)
storage.mode(full2)<-"numeric"
full_matrix <- xgb.DMatrix(data = full2[1:nr,], label = train$y)
depth=as.matrix(c(2,3,4))
etaF=as.matrix(c(0.3,0.5,0.7,0.9))
oosrmse=array(1,c(nrow(depth),nrow(etaF)))
oosindex=array(1,c(nrow(depth),nrow(etaF)))
for (j in 1:nrow(depth))
{ 
  for (k in 1:nrow(etaF))
  {
    bst_f <- xgb.cv(data = full_matrix, max.depth = depth[j], eta = etaF[k], nthread = 2, nround = 8,  nfold=20, objective = "reg:linear")
    oosrmse[j,k]=min(bst_f$evaluation_log$test_rmse_mean)
    oosindex[j,k]=which((bst_f$evaluation_log$test_rmse_mean==min(bst_f$evaluation_log$test_rmse_mean)))
  }
}

bstF <- xgboost(data = full_matrix, max.depth = 4, eta = 0.95, nthread = 2, nround = 3, objective = "reg:linear")
names = dimnames(full_matrix)[[2]]
importance_matrix = xgb.importance(names, model=bstF)
gp = xgb.plot.importance(importance_matrix[1:15,])

# test_matrix <- xgb.DMatrix(data = test2)
test_full_matrix <- xgb.DMatrix(data = full2[(nr+1):nrow(full2),])
pred <- predict(bstF, test_full_matrix)
sub.file = data.frame(ID = test$ID, y = pred)
#sub.file = aggregate(data.frame(cost = sub.file$cost), by = list(id = sub.file$id), mean)
write.csv(sub.file, "submit.csv", row.names = FALSE, quote = FALSE)
