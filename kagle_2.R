##The aim of this work to try to elaborate an application credit scoring without using the parameters in the dataset which related with customer behavior or external risk assesment. We will use logistic regression and RPA model.

# As first step, import dataset (https://www.kaggle.com/datasets/zaurbegiev/my-dataset) and necessary libraries.

install.packages('gmodels')
install.packages("tables")
install.packages("caret")
install.packages("corrplot")
install.packages("woeBinning")
install.packages("C50") 
install.packages("ROCit")
install.packages("mice")


library(gmodels)
library(plyr)
library(caret)
library(corrplot)
library(woeBinning)
library(C50)
library(ROCit)
library(mice)

dat <- read.csv("blsd_train.csv", na.strings=c(""," ", "NA"))
dat <- as.data.frame(dat)


#Abandom of variables which are not in our scope (external rating, credit history, etc.) 'Loan.Status' is our target variable.

dat_r <- dat[ , c(3:5, 7:12, 17)]
dat<-dat_r
dat$Loan.Status <- as.factor(dat$Loan.Status)

#Data preparation: handling the missing values, outliers an other problems in our dataset (wrong or dummy inputs)

colSums(is.na(dat))
md.pattern(dat)
#There are 513 rows which many missing values -  dropping them
data_subset <- dat[ , c("Loan.Status", "Maximum.Open.Credit")]  
dat <- dat[complete.cases(data_subset), ]

#Current.Loan amonunt - replacing wrong inputs with average
foravg <- dat[dat$Current.Loan.Amount != 99999999, ]
mean(foravg$Current.Loan.Amount, na.rm = TRUE)
dat["Current.Loan.Amount"][dat["Current.Loan.Amount"]== 99999999] <- mean(foravg$Current.Loan.Amount, na.rm = TRUE)

#Annual Income - replacing missing values with average
dat$Annual.Income[is.na(dat$Annual.Income)] <- mean(dat$Annual.Income, na.rm = TRUE) 

##Years in current Job - changing the values
nv <- c('< 1 year' = 0, '1 year' = 1, '2 years' = 2, '3 years' = 3, '4 years' =4, '5 years'= 5, "6 years" = 6, '7 years' = 7, '8 years' = 8, "9 years" =9, '10+ years'= 10)
dat$Years.in.current.job <- as.integer(revalue(dat$Years.in.current.job,nv))

#Search for and drop outliers
boxplot(dat$Annual.Income)
hist(dat$Annual.Income)
dat2 <- dat[dat$Annual.Income < 3000000 , ] #
dat <- dat2

boxplot(dat$Monthly.Debt)
hist(dat$Monthly.Debt)
dat2 <- dat[dat$Monthly.Debt < 80000 & dat$Monthly.Debt !=0, ]
dat <- dat2

boxplot(dat$Maximum.Open.Credit)
dat2 <- dat[dat$Maximum.Open.Credit < 4000000 , ]
dat <- dat2

#Checking extreme values in dept to income
dat$DtI <- (dat$Annual.Income/12/dat$Monthly.Debt)
summary(dat)
hist(dat$DtI)
md <- dat[dat$DtI > 1000 , ]
dat2 <-dat[dat$DtI<1000 , ] 
dat<- dat2

#Checking the correlation between our variables. To strong correlation between the variables deteriorates the model efficiency.

dat_cor <- dat[ 1:10, c( "Current.Loan.Amount", "Annual.Income", "Monthly.Debt" , "Years.of.Credit.History", "Maximum.Open.Credit")]
corrplot(cor(dat_cor),tl.cex = 1, method = "color",  addCoef.col="black", number.cex = 1)

#WOE transformation and checking Information Value

dat <- na.omit(dat)
dat_woe <- dat
dat_woe$Loan.Status <- ifelse(dat_woe$Loan.Status == "Charged Off",1,0)

dat_woe$Term <- as.factor(dat$Term)
dat_woe$Home.Ownership <- as.factor(dat$Home.Ownership)
dat_woe$Purpose <- as.factor(dat$Purpose)
dat_woe <- dat_woe [ , -11]

dat_woe <- na.omit(dat_woe)

woe_er <- woe.binning(dat_woe, 'Loan.Status', dat_woe)
dat_woe_kieg <- woe.binning.deploy(dat_woe, woe_er, add.woe.or.dum.var='woe')
dat_woe_kieg <- as.data.frame(dat_woe_kieg)
woe_table <- woe.binning.table(woe_er)

woe.binning.plot(woe_er) 
woe.binning.plot(woe_er, plot.range='5:5')

woerules <- woe.binning(df = dat_woe_kieg,target.var = "Loan.Status",pred.var = dat_woe_kieg,event.class = 1)

#Unfortunately the information values of these variables are very weak; we can not expect a robust, effective model. :-( Let's comtinue anyway... with a logostuc regression model

#Logistic regresson model - using WOE transformation

dat_l <- dat_woe_kieg[ , c("Loan.Status","woe.Current.Loan.Amount.binned", "woe.Term.binned", "woe.Annual.Income.binned",   "woe.Purpose.binned",  "woe.Years.of.Credit.History.binned", "woe.Monthly.Debt.binned", "woe.Maximum.Open.Credit.binned", "woe.Years.in.current.job.binned")]

set.seed(2000)
g<- runif(nrow(dat_l))
dat_l <- dat_l[order(g),]

train <- dat_l[1:54000, 1:9]
test <- dat_l[54001:nrow(dat_l) , 2:9] 

test_t <- dat_l[54001:nrow(dat_l), ] #test data with target variable

lreg <- glm(Loan.Status~., family = binomial, data = train)
step(lreg) # boosting the model

log_predict <- predict(lreg, type = "response", newdata = test[,])


# Using probability cutoff of 70%.
pred_risk <- factor(ifelse(log_predict >= 0.30, "Fully Paid", "Charged Off"))
actual_risk <- factor(ifelse(test_t$Risk=='1',"bad", "good"))


##Validation and evaluation of the model with AUROC

table(actual_risk,pred_risk)
CrossTable(actual_risk, pred_risk)
CrossTable(test_teljes$Loan.Status, pred_risk, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual', 'predicted'))
confusionMatrix(reference=actual_risk, data=pred_risk, mode = "everything") 

#### ROCit
actual_risk <- factor(test_t$Loan.Status)
pred_risk_lreg <- as.numeric(log_predict)
ROCit_obj <- rocit(score=pred_risk_lreg,class=actual_risk)
plot(ROCit_obj)
summary(ROCit_obj)

#RPA model - without WOE transformation

#train and test data separation 
set.seed(4000)
in_train <- sample(1:nrow(dat), size = 59000)
rpa_train <- dat[ in_train,]
test_data  <- dat[-in_train,] 

#Predictive variables 
vars2 <- c("Monthly.Debt", "Annual.Income", "Years.in.current.job",  "Term", "Home.Ownership", "Purpose", "Years.of.Credit.History", "Current.Loan.Amount")

#train our model using C5.0 function; check the result
tree_mod <- C5.0(x = rpa_train[ , vars2], y = rpa_train$Loan.Status, trials = 50)
tree_mod
plot(tree_mod)
summary(tree_mod)
C5imp(tree_mod)

#prediction on test data
pred_dat <- predict(tree_mod, test_data)

#possible to investigate the probbality of default 
pred_dat_df_pd <- predict(tree_mod, newdata=test_data[1:20 , ], type = "prob")

pred_dat_df_dfpd <- as.data.frame(pred_dat_df_pd)
head(pred_dat_df_dfpd)

## boosting with cost matrix - the cost of FN (fefault predicted as good) is more expensive for us
cost_matrix <- matrix(c(0, 1, 2, 0), nrow = 2)
rownames(cost_matrix) <- colnames(cost_matrix) <- c("Charged.Off", "Fully.Paid")
cost_matrix

tree_mod_cost  <- C5.0(x = rpa_train[ , vars2], y = rpa_train$Loan.Status, trials = 10, costs = cost_matrix)
summary(tree_mod_cost)

#validation on test data - confusion matrix and AUROC
CrossTable(test_data$Loan.Status, pred_dat, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual', 'predicted'))
confusionMatrix(reference=pred_dat, data=test_data$Loan.Status, mode = "everything") 

pred_risk_rpa <- as.numeric(pred_dat)

ROCit_objrpa <- rocit(score=pred_risk_rpa,class=test_data$Loan.Status)
plot(ROCit_objrpa)
summary(ROCit_objrpa)
