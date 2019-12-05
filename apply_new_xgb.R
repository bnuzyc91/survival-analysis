install.packages("shiny")
library("shiny")

library(xgboost)
library("survival")
library("randomForestSRC")

set.seed(1234)
data(pbc,package="randomForestSRC")
pbc<-pbc[complete.cases(pbc),]

ind<-pbc$status==0
pbc$days_model<-pbc$days
pbc$days_model[ind]<--1*pbc$days[ind]

head(pbc)

dtrain<-xgb.DMatrix(data.matrix(pbc[,c(seq(2,17))]),label=pbc$days_model)
dtest<-xgb.DMatrix(data.matrix(pbc[,c(seq(2,17))]),label=pbc$days_model)


bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, 
                      nrounds = 100, objective = "survival:cox")

pred <- predict(bstDMatrix, dtest,output_margin = False)

# size of the prediction vector
print(length(pred))

for (i in seq(1,length(pred))){
  tmp=i+1
  for (j in seq(tmp,length(pred))){
    
    if (pbc$status[i]==1 & pbc$status[j]==1){total<-total+1}
    if (pbc$status[i]==1 & pbc$status[j]==1 &  
     pred[i]>pred[j] & (pbc$days[i]>pbc$days[j])){
       cpairs<-cpairs+1
    }
    else if (pbc$status[i]==1 & pbc$status[j]==1 &  
             pred[i]<pred[j] & pbc$days[i]<pbc$days[j]){
      cpairs<-cpairs+1
    }
    
  }
  
}

options(scipen=999)
  cindex<- cpairs/total 
  1-cindex
  
