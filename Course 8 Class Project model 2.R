### This code was used for pre-processing and data prep for Course 8 in the Johns Hopkins Data Science SPecialization  Final Project.
## Prepared by M. Amati April 27 2018.
## The R objects loaded here were created in the Couse 8 Class Project prep.R file.

library(caret)
library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(dplyr)
library(scales)

# Set the working directory where the project is stored.
Dir<-"C:/Users/Mike/Documents/Coursera/JohnsHopkinsDataScience/Course_8_Machine_Learning"
#C:\Users\Mike\Documents\Coursera\JohnsHopkinsDataScience\Course_8_Machine Learning
setwd(Dir)
########################################################################################################

##Load needed objects
train_all<-readRDS("train_all.rds")
train_sig<-readRDS("train_sig.rds")
test_full<-readRDS("test_full.rds")
traindata = train_all

#initialize Training Accuracy Matrix
Training_Accuracy<- matrix(data=0,nrow=4,ncol=3)

###Begin Modeling 
## Note that the fit portions are commented out after success so they do not need to be rerun in subsequent calls.   To recreate them uncomment and rerun the train steps.  

###############   Full Training DAta #######
#Random Forest 
#RF1Fit <- train(classe ~.,data= traindata, method="rf")
#saveRDS(RF1Fit, file = "RF1FIT.rds")
RF1Fit<-readRDS("RF1FIT.rds")
RF1Pred_Train <- predict(RF1Fit, traindata)
Training_Accuracy[1,1]<-confusionMatrix(traindata$classe, RF1Pred_Train)$overall[1]

#GBM
#GBM1Fit <- train(classe ~.,data= traindata, method="gbm",verbose=FALSE)
#saveRDS(GBM1Fit, file = "GBM1FIT.rds")
GBM1Fit<-readRDS("GBM1FIT.rds")
GBM1Pred_Train <- predict(GBM1Fit, traindata)
Training_Accuracy[2,1]<-confusionMatrix(traindata$classe, GBM1Pred_Train)$overall[1]

#LDA
#LDA1Fit <- train(classe ~.,data= traindata, method="lda")
#saveRDS(LDA1Fit, file = "LDA1FIT.rds")
LDA1Fit<-readRDS("LDA1FIT.rds")
LDA1Pred_Train <- predict(LDA1Fit, traindata)
Training_Accuracy[3,1]<-confusionMatrix(traindata$classe, LDA1Pred_Train)$overall[1]

#CART
#CART1Fit <- train(classe ~.,data= traindata, method="rpart")
#saveRDS(CART1Fit, file = "CART1FIT.rds")
CART1Fit<-readRDS("CART1Fit.rds")
CART1Pred_Train <- predict(CART1Fit, traindata)
Training_Accuracy[4,1]<-confusionMatrix(traindata$classe, CART1Pred_Train)$overall[1]

########################################################################################

###################################################################
### full set with PCA
preProcfull <- preProcess(traindata,method="pca",thresh = 0.8)
trainPCfull <- predict(preProcfull,traindata)

#RF
#RF2Fit <- train(classe ~.,data= trainPCfull, method="rf")
#saveRDS(RF2Fit, file = "RF2FIT.rds")
RF2Fit<-readRDS("RF2FIT.rds")
RF2Pred_Train <- predict(RF2Fit, trainPCfull)
Training_Accuracy[1,2]<-confusionMatrix(traindata$classe, RF2Pred_Train)$overall[1]

#GBM
#GBM2Fit <- train(classe ~.,data= trainPCfull, method="gbm",verbose=FALSE)
#saveRDS(GBM2Fit, file = "GBM2FIT.rds")
GBM2Fit<-readRDS("GBM2FIT.rds")
GBM2Pred_Train <- predict(GBM2Fit, trainPCfull)
Training_Accuracy[2,2]<-confusionMatrix(traindata$classe, GBM2Pred_Train)$overall[1]


#LDA
#LDA2Fit <- train(classe ~.,data= trainPCfull, method="lda")
#saveRDS(LDA2Fit, file = "LDA2FIT.rds")
LDA2Fit<-readRDS("LDA2FIT.rds")
LDA2Pred_Train <- predict(LDA2Fit, trainPCfull)
Training_Accuracy[3,2]<-confusionMatrix(traindata$classe, LDA2Pred_Train)$overall[1]

#CART
#CART2Fit <- train(classe ~.,data= trainPCfull, method="rpart")
#saveRDS(CART2Fit, file = "CART2Fit.rds")
CART2Fit<-readRDS("CART2Fit.rds")
CART2Pred_Train <- predict(CART2Fit, trainPCfull)
Training_Accuracy[4,2]<-confusionMatrix(traindata$classe, CART2Pred_Train)$overall[1]
#################################################################################################

############### Significant with PCA
trainsdata = train_sig


preProcsig <- preProcess(trainsdata,method="pca",thresh = 0.8)
trainPCsig <- predict(preProcsig,trainsdata)

#RF
#RF3Fit <- train(classe ~.,data= trainPCsig, method="rf")
#saveRDS(RF3Fit, file = "RF3FIT.rds")
RF3Fit<-readRDS("RF3FIT.rds")
RF3Pred_Train <- predict(RF3Fit, trainPCsig)
Training_Accuracy[1,3]<-confusionMatrix(trainsdata$classe, RF3Pred_Train)$overall[1]


#GBM
#GBM3Fit <- train(classe ~.,data= trainPCsig, method="gbm",verbose=FALSE)
#saveRDS(GBM3Fit, file = "GBM3FIT.rds")
GBM3Fit<-readRDS("GBM3FIT.rds")
GBM3Pred_Train <- predict(GBM3Fit, trainPCsig)
Training_Accuracy[2,3]<-confusionMatrix(trainsdata$classe, GBM3Pred_Train)$overall[1]


#LDA
#LDA3Fit <- train(classe ~.,data= trainPCsig, method="lda")
#saveRDS(LDA3Fit, file = "LDA3FIT.rds")
LDA3Fit<-readRDS("LDA3FIT.rds")
LDA3Pred_Train <- predict(LDA3Fit, trainPCsig)
Training_Accuracy[3,3]<-confusionMatrix(trainsdata$classe, LDA3Pred_Train)$overall[1]

#CART
#CART3Fit <- train(classe ~.,data= trainPCsig, method="rpart")
#saveRDS(CART3Fit, file = "CART3Fit.rds")
CART3Fit<-readRDS("CART3Fit.rds")
CART3Pred_Train <- predict(CART3Fit, trainPCsig)
Training_Accuracy[4,3]<-confusionMatrix(trainsdata$classe, CART3Pred_Train)$overall[1]

## Finalize and save Training Accuracy 
Training_Accuracy
colnames(Training_Accuracy)<-c("FULL","FULL PCA","SIGNIFICANT PCA")
rownames(Training_Accuracy)<-c("Random Forest","Gradient Boosting","Linear Discriminant Analysis","CART")
Training_Accuracy<-round(Training_Accuracy,2)
saveRDS(Training_Accuracy,file="Training_Accuracy.rds")


#################  Testing
#Initialize Training Accuracy
Testing_Accuracy<- matrix(data=0,nrow=4,ncol=3)
#Create test_full
test_all<-test_full[,names(test_full) %in% names(train_all)]
#create test_sig
test_sig<-test_full[,names(test_full) %in% names(train_sig)]
#Create testlPCfull
testPCfull <- predict(preProcfull,test_all)
#Create testPCsig
testPCsig <- predict(preProcsig,test_sig)

#run tests

RF1Pred_Test <- predict(RF1Fit, test_all)
RF2Pred_Test <- predict(RF2Fit, testPCfull)
RF3Pred_Test <- predict(RF3Fit, testPCsig)
Testing_Accuracy[1,1]<-confusionMatrix(test_all$classe, RF1Pred_Test)$overall[1]
Testing_Accuracy[1,2]<-confusionMatrix(test_all$classe, RF2Pred_Test)$overall[1]
Testing_Accuracy[1,3]<-confusionMatrix(test_sig$classe, RF3Pred_Test)$overall[1]

GBM1Pred_Test <- predict(GBM1Fit, test_all)
GBM2Pred_Test <- predict(GBM2Fit, testPCfull)
GBM3Pred_Test <- predict(GBM3Fit, testPCsig)
Testing_Accuracy[2,1]<-confusionMatrix(test_all$classe, GBM1Pred_Test)$overall[1]
Testing_Accuracy[2,2]<-confusionMatrix(test_all$classe, GBM2Pred_Test)$overall[1]
Testing_Accuracy[2,3]<-confusionMatrix(test_sig$classe, GBM3Pred_Test)$overall[1]

LDA1Pred_Test <- predict(LDA1Fit, test_all)
LDA2Pred_Test <- predict(LDA2Fit, testPCfull)
LDA3Pred_Test <- predict(LDA3Fit, testPCsig)
Testing_Accuracy[3,1]<-confusionMatrix(test_all$classe, LDA1Pred_Test)$overall[1]
Testing_Accuracy[3,2]<-confusionMatrix(test_all$classe, LDA2Pred_Test)$overall[1]
Testing_Accuracy[3,3]<-confusionMatrix(test_sig$classe, LDA3Pred_Test)$overall[1]

CART1Pred_Test <- predict(CART1Fit, test_all)
CART2Pred_Test <- predict(CART2Fit, testPCfull)
CART3Pred_Test <- predict(CART3Fit, testPCsig)
Testing_Accuracy[4,1]<-confusionMatrix(test_all$classe, CART1Pred_Test)$overall[1]
Testing_Accuracy[4,2]<-confusionMatrix(test_all$classe, CART2Pred_Test)$overall[1]
Testing_Accuracy[4,3]<-confusionMatrix(test_sig$classe, CART3Pred_Test)$overall[1]

Testing_Accuracy
colnames(Testing_Accuracy)<-c("FULL","FULL PCA","SIGNIFICANT PCA")
rownames(Testing_Accuracy)<-c("Random Forest","Gradient Boosting","Linear Discriminant Analysis","CART")
Testing_Accuracy<-round(Testing_Accuracy,2)
saveRDS(Testing_Accuracy,file="Testing_Accuracy.rds")

### Validation

validation_full<-readRDS("validation_full.rds")
### Validation
val_all<-validation_full[,names(validation_full) %in% names(train_all)]
#create test_sig
val_sig<-validation_full[,names(validation_full) %in% names(train_sig)]
#Create testlPCfull
valPCfull <- predict(preProcfull,val_all)
#Create testPCsig
valPCsig <- predict(preProcsig,val_sig)

RF1Pred_Val <- predict(RF1Fit, val_all)
Val_CM<-confusionMatrix(val_all$classe, RF1Pred_Val)
Validation_Accuracy<-confusionMatrix(val_all$classe, RF1Pred_Val)$overall[1]
saveRDS(Validation_Accuracy,file="Validation_Accuracy.rds")
###  Assign Values for Quiz
TEST<-readRDS("TEST.RDS")
TEST_all<-TEST[,names(TEST) %in% names(train_all)]
RF1Pred_Quiz<-predict(RF1Fit,TEST_all)