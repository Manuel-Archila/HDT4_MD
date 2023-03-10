# instalación de paquetes -------------------------------------------------

# install.packages("tree")

library(rpart)
library(caret)
library(tree)
library(rpart.plot)
library(randomForest)

datos <- iris

# variable respuesta la clase de la flor
porciento <- 70/100

set.seed(123)

trainRowsNumber<-sample(1:nrow(datos),porciento*nrow(datos))
train<-datos[trainRowsNumber,]
test<-datos[-trainRowsNumber,]
arbolModelo<-rpart(Species~.,datos,method = "class")
rpart.plot(arbolModelo)

save(train,test,arbolModelo, file = "Variables.RData")
load("Variables.RData")
dt_model<-rpart(Species~.,train,method = "class")
plot(dt_model);text(dt_model)
prp(dt_model)
rpart.plot(dt_model)

head(test)
prediccion <- predict(dt_model, newdata = test[1:4])

#Apply: Para cada fila, determina el nombre de la columna del valor máximo entre los tres valores de una fila
columnaMasAlta<-apply(prediccion, 1, function(x) colnames(prediccion)[which.max(x)])
test$prediccion<-columnaMasAlta #Se le añade al grupo de prueba el valor de la predicción

cfm<-confusionMatrix(as.factor(test$prediccion),test$Species)
cfm

#con caret
ct<-trainControl(method = "cv",train[,1:4],number=10, verboseIter=T)
modelorf<-train(Species~.,data=train,method="rpart",trControl = ct)
prediccionADVC<-predict(modelorf,newdata = test[,1:4])
test$predADVC<-prediccionADVC
cfmCaret <- confusionMatrix(test$predADVC,test$Species)



#con caret
ct<-trainControl(method = "cv",train[,1:4],number=10, verboseIter=T)
modelorf<-train(Species~.,data=train,method="rf",trControl = ct)
prediccionrfVC<-predict(modelorf,newdata = test[,1:4])
test$predrfVC<-prediccionrfVC
cfmCaret <- confusionMatrix(test$predrfVC,test$Species)
#con random forest
modeloRF1<-randomForest(Species~.,data=train)
prediccionRF1<-predict(modeloRF1,newdata = test[,1:4])
testCompleto<-test
testCompleto$predRF<-prediccionRF1
cfmRandomForest <- confusionMatrix(testCompleto$predRF, testCompleto$Species)
