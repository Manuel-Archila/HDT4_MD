library(tidyr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(randomForest)
library(mlr) #machine learning in r


titanic <- read.csv("Datos/titanic.csv")
nrow(titanic)
ncol(titanic)
colnames(titanic)
titanic<-titanic[,c("Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked","Survived")]
str(titanic)
titanic[titanic$Cabin == "","Cabin"]<-"NO CABIN"
titanic$Cabin<-gsub(" ","_",titanic$Cabin)

titanic <- titanic %>% mutate_at(c("Pclass","Sex","Parch","Cabin","Embarked","Survived"),as.factor)

set.seed(456)
porciento <- 0.7
filas_train<-sample(1:nrow(titanic),nrow(titanic)*porciento)
train <- titanic[filas_train,]
test <- titanic[-filas_train,]

nrow(train)
nrow(test)

modelo1<-rpart(Survived~.,train, method = "class")
rpart.plot(modelo1)

y <- test[,9]
test<-test[,-9]

colnames(test)
ypred <- predict(modelo1,newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)

confusionMatrix(ypred,y)


table(train$Survived)
table(y)
table(titanic$Survived)

#Muestreo estratificado
sobrevivientes<-titanic[titanic$Survived == 1,]
muertos<-titanic[titanic$Survived == 0,]
 
set.seed(456)
filas_train_1<-sample(1:nrow(sobrevivientes),nrow(sobrevivientes)*0.7)
filas_train_0<-sample(1:nrow(muertos),nrow(muertos)*0.7)

train<-rbind(muertos[filas_train_0,],sobrevivientes[filas_train_1,])
test<-rbind(muertos[-filas_train_0,],sobrevivientes[-filas_train_1,])

y <- test[,"Survived"]
test<-test[,-9]

modelo2<-rpart(Survived~.,train,method = "class")
rpart.plot(modelo2)

ypred <- predict(modelo2, newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)

confusionMatrix(ypred,y)


# Cross Validation

ct<-trainControl(method = "cv",number=10, verboseIter=T)
modelo3 <- train(train[,-9],train$Survived, trControl = ct, method="rpart")

ypred <- predict(modelo3, newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)
confusionMatrix(ypred,y)


#Cambiando el maxdepth
modelo4<- rpart(Survived~.,train,method = "class",maxdepth=3)
rpart.plot(modelo4)

ypred <- predict(modelo4, newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)
confusionMatrix(ypred,y)

modelo5<- rpart(Survived~.,train,method = "class",maxdepth=10)
rpart.plot(modelo5)

ypred <- predict(modelo5, newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)
confusionMatrix(ypred,y)

modelo6<- rpart(Survived~.,train,method = "class",maxdepth=2)
rpart.plot(modelo6)

ypred <- predict(modelo6, newdata = test)
ypred<-apply(ypred, 1, function(x) colnames(ypred)[which.max(x)])
ypred <- factor(ypred)
confusionMatrix(ypred,y)

#variable<-tune(rpart,train[,-9],train$Survived, range=list(maxdepth=2:6))
colnames(train)

y <- test[,"Survived"]
test<-test[,c(-7,-9)]

modeloRF <- randomForest(Survived~.,train[,-7],na.action = na.omit)
ypred <- predict(modeloRF,newdata = test)
ypred <- factor(ypred)
confusionMatrix(ypred,y)

#Tuneando hiperparámetros con mlr
#¿Qué parametros se pueden tunear en un árbol de decisión?
#Eliminamos Cabin
train<- train[,-7]
test<- test[,-7]
getParamSet("classif.rpart")
clasificador <- makeClassifTask(data=train, target = "Survived")
tablaParametros<-makeParamSet(makeDiscreteParam("maxdepth",values=1:15))
controlGrid <- makeTuneControlGrid()
cv <- makeResampleDesc("CV",iters=3L)
metrica <- acc
set.seed(456)
dt_tuneparam <- tuneParams(learner = "classif.rpart",
                           task = clasificador,
                           resampling = cv,
                           measures = metrica,
                           par.set=tablaParametros,
                           control=controlGrid,
                           show.info=T)
result_hyperparam <- generateHyperParsEffectData(dt_tuneparam, partial.dep = TRUE)

ggplot(
  data = result_hyperparam$data,
  aes(x = maxdepth, y=acc.test.mean)
) + geom_line(color = 'darkblue')

best_parameters = setHyperPars(
  makeLearner("classif.rpart"), 
  par.vals = dt_tuneparam$x
)

best_model = train(best_parameters, clasificador)
test <- titanic[-filas_train,]

d.tree.mlr.test <- makeClassifTask(
  data=test, 
  target="Survived"
)
results <- predict(best_model, task = d.tree.mlr.test)$data
confusionMatrix(results$truth, results$response)
