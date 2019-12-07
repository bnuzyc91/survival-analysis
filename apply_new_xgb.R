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
  
require(xgboost)
library("survival")
library("randomForestSRC")
set.seed(1234)
options(scipen=999)
data(pbc, package = "randomForestSRC")
pbc_orig <- pbc[complete.cases(pbc),]
pbc <- pbc[complete.cases(pbc),]

head(pbc)
#days_model is the event/censor date for cox regression ; negative value=censored, positive value=event
ind<-pbc$status==0
pbc$days_model<-pbc$days
pbc$days_model[ind]<--1*pbc$days[ind]

276*0.8
trainind<-seq(1:220)
dtrain<-xgb.DMatrix(data.matrix(pbc[c(1:220),c(seq(4,19))]),label=pbc$days_model[c(1:220)])
dtest<-xgb.DMatrix(data.matrix(pbc[c(221:276),c(seq(4,19))]),label=pbc$days_model[c(221:276)])

watchlist <- list(train = dtrain, test = dtest)
bstDMatrix <- xgboost(data = dtrain, max.depth = 6, eta = 0.1, nthread = 2, 
                      nrounds = 1000, objective = "survival:cox")

preds <- predict(bstDMatrix, dtest)
hist(preds)

origscore<-bstDMatrix$best_score
origiter<-bstDMatrix$best_iteration
origscore



# size of the prediction vector


mycindex<-function(preds){
  permissible<-1
  concordance<-1
  endind=length(preds)-1
  for (i in seq(1,endind)){
    tmp=i+1
    for (j in seq(tmp,length(preds))){
      
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==0) & (pbc$status[j]==0)){ next } 
      if((pbc$days[i]<pbc$days[j]) & (pbc$status[i]==0) ){ next } 
      if((pbc$days[j]<pbc$days[i]) & (pbc$status[j]==0) ){ next } 
      
      permissible<-permissible+1
      
      if (pbc$status[i]==1 & pbc$status[j]==1 &  
          preds[i]>preds[j] & (pbc$days[i]>pbc$days[j])){
        concordance<-concordance+1
        
        #com_value<-c(concordance,preds[i], preds[j], pbc$days[i], pbc$days[j])
        #print (com_value)
      }
      if (pbc$status[i]==1 & pbc$status[j]==1 &  
               preds[i]<preds[j] & pbc$days[i]<pbc$days[j]){
        concordance<-concordance+1
       
        com_value<-c(concordance,preds[i], preds[j], pbc$days[i], pbc$days[j])
        print (com_value)
      }
      
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==1) & (pbc$status[j]==1) & (preds[i]!=preds[j]))
      {
        concordance<-concordance+0.5
        
      }
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==1) & (pbc$status[j]==0) &  (preds[i]<preds[j]))
      {
        concordance<-concordance+1
        
      }
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==0) & (pbc$status[j]==1) & (preds[i]>preds[j]))
      {
        concordance<-concordance+1
        
      }
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==1) & (pbc$status[j]==0) & (preds[i]>=preds[j]))
      {
        concordance<-concordance+0.5
       
      }
      if((pbc$days[i]==pbc$days[j]) & (pbc$status[i]==0) & (pbc$status[j]==1) & (preds[i]<=preds[j])) {
        concordance<-concordance+0.5
        
      }
    }
    
  }
  cindex<- concordance/permissible
  myout<-c(concordance,permissible,cindex)
  return (myout)
}


cindex_xgb<-mycindex(preds)
cindex_xgb
# add bunch of code permissible: 6106->6111 concordance 96->98.5

importance <-TRUE
ntree<-500
partial1 <- rfsrc(Surv(days, status) ~ .,
                  data = pbc_orig[c(1:220),],
                  ntree=ntree,nodedepth =3,
                  importance = importance,tree.err=FALSE)
partial1.pred <- predict(partial1, pbc_orig[c(221:276),])
1-partial1.pred$err.rate[ntree]
#partial1.pred$predicted
cindex_rsf<-mycindex(partial1.pred$predicted)
cindex_rsf

pbc2<-read.csv("/home/s792321/xgb_new/pbc.csv")
pbc2<-pbc2[,-1]
importance <-TRUE
ntree<-500
partial2 <- rfsrc(Surv(days, status) ~ .,
                  data = pbc2[c(1:220),],
                  ntree=ntree,nodedepth =3,
                  importance = importance,tree.err=FALSE)
partial2.pred <- predict(partial1, pbc2[c(221:276),])
1-partial2.pred$err.rate[ntree]
