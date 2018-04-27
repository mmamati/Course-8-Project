### This code was used for pre-processing and data prep for Course 8 in the JOhns Hopkins Data Science SPecialization  Final Project.
## Prepared by M. Amati April 27 2018.
## Modeling development is done in the Couse 8 Class Project model 2.R file.

library(caret)
library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(dplyr)
###############################################################################################
####  STEP 1 LOAD DATA AND SPLIT INTO TRAINING, TEST, VALIDATION ####

# Set the working directory where the project is stored.
Dir<-"C:/Users/Mike/Documents/Coursera/JohnsHopkinsDataScience/Course_8_Machine_Learning"
#C:\Users\Mike\Documents\Coursera\JohnsHopkinsDataScience\Course_8_Machine Learning
setwd(Dir)
#set the link to the data
trainfileURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testfileURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#set the name of the outfiles
trainfile<-paste(Dir,"pml_training.csv",sep="/")
testfile<-paste(Dir,"pml_testing.csv",sep="/")
#download the data
download.file(trainfileURL,trainfile)
download.file(testfileURL,testfile)
#Read with csv.  
# Captilalized are the full sets as provided.  Will split TRAIN further.
TRAIN<-read.csv("pml_training.csv")
TEST<-read.csv("pml_testing.csv")



#Split into training, test, and validation 70/20/10 split
inTrain = createDataPartition(TRAIN$classe, p = 0.7)[[1]]
train_full = TRAIN[inTrain,]
testtemp = TRAIN[-inTrain,]
inTest = createDataPartition(testtemp$classe, p = 2/3)[[1]]
test_full=testtemp[inTest,]
validation_full=testtemp[-inTest,]
###############  End Step 1 Load Data and split into training, test, and validation #####################################################

###############################################################################################################################
### STEP 2 - REDUCE TO VARIABLES WITH SUFFICIENT COVERAGE
#removing variables with very little population


retain<-c()
exclude<-NULL
cutoff<-1000
#Loop over full training set and retain if there are few enough missing and empty cells
for (i in 1:length(names(train_full)))
{ p<-sum(is.na(train_full[,i]))
  q<-sum(train_full[,i]!="",na.rm=TRUE)
  if (p<cutoff&q>cutoff)
     {retain<-c(retain,i)}
  else
     {exclude<-c(exclude,i)}

}

#train is the full training set that will be used.  It has 60 columns, but some won't be dependent variables
# X,user_name,raw_timestamp_part_1,raw_timestamp_part2, cvtd_timestamp, new_window, num_window will not be used
train<-train_full[,retain]

#######  End Step 2 
#######################################################################################################################

#####################################################################################################################
#### Step 3 - Check correlation matrix.  How many pairs are highly correlated


#from dyplr, select numeric fields only.
forcor<-select_if(train, is.numeric)
#calculate correlation matrix
cormat<-as.matrix(cor(forcor,method="pearson"))
#check how many pairs are over a threshold.  subtract out 1s for the diagonals and divide by 2 to recognize they are duplicated
cor_cut=0.5
numhighcor<-length(which((cormat>cor_cut|cormat<(-cor_cut))&cormat!=1))/2

#There are 92 pairs of variables highly correlated at a threshold of 0.5  

#### End Step 3
##################################################################################################################

############  Step 4 - Reduce to variables with predictive power on their own######################################################

#calculate the mean by class and rearrange to make useful.
DF<-as.data.frame(t(aggregate(train,by=list(train$classe),FUN=mean)))
DF<-t(aggregate(train,by=list(train$classe),FUN=mean))
colnames(DF)<-c("A","B","C","D","E")
DF<-as.data.frame(DF)
DF$A<-as.numeric(as.character(DF$A))
DF$B<-as.numeric(as.character(DF$B))
DF$C<-as.numeric(as.character(DF$C))
DF$D<-as.numeric(as.character(DF$D))
DF$E<-as.numeric(as.character(DF$E))
#filter out variables that won't be used for predictors
DG<-DF[complete.cases(DF),]
DG<-DG[-c(1:4),]

#determine the classe with the minimum and maximum average for each potential predictor
minmax<-data.frame(rownames(DG),colnames(DG)[apply(DG,1,which.min)],colnames(DG)[apply(DG,1,which.max)])
colnames(minmax)<-c("measurement","Min Class","Max Class")

#for each of the 52 predictors, run a t test to see if the means from the minimum and maximum class are different
#use a fairly high threshhold for significance
Pcut<-0.01
significant<-NULL

for (i in 1:nrow(minmax)) {
minclasse<-minmax$`Min Class`[i]
maxclasse<-minmax$`Max Class`[i]
maxv<-train[train$classe==maxclasse,i+7]
minv<-train[train$classe==minclasse,i+7]
p<-t.test(maxv,minv,alternative="t")$p.value
#print(p)
if (p<Pcut) {significant<-c(significant,i+7)}
}

## Reduce training set to those that are considered significant and add in class
train_sig<-train[,c(significant,60)]
train_all<-train[,8:60]
########  End step 4
########################################################################################################################

saveRDS(train_all, file = "train_all.rds")
saveRDS(train_sig ,file = "train_sig.rds")
saveRDS(TRAIN, file = "TRAIN.rds")
saveRDS(TEST, file = "TEST.rds")
saveRDS(test_full, file = "test_full.rds" )
saveRDS(validation_full, file = "validation_full.rds")
########################################################################################################
