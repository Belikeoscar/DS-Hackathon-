# The essence of this is to predict diagnosis

library('caret')

# Select all rows and the first 32 columns
data <- cancer_dataset[,c(1:32)]

# Use createDataPartition function to create a training and test set (70/30 split)
partition <- createDataPartition(data$diagnosis, p=0.7, list=FALSE)

trainingData <- data[partition,]
testData <- data[-partition,]

trainingData$diagnosis <- as.factor(trainingData$diagnosis)
testData$diagnosis <- as.factor(testData$diagnosis)

# Now build a model using K-NN with caret using the training set:
model <- train(diagnosis~., method="knn", data = trainingData)

model

# Now test the model on the test set:
testingTheModel <- predict(model, newdata = testData)

?predict

# Use the confusion matrix function to study the various evaluation metrics:
confusionMatrix(testingTheModel, testData$diagnosis)

# The following code will also tell use what the no-information rate is:
table(testData$diagnosis)

NIR = majorityClass / numberOfData
# So B is the majority class, hence the no information rate is 107/170 (~62.94%)

# to solve the randomness of machine learning, you have to set seed

# Try the following statistical test to get a p-value when comparing proportion of correctly
# classified cases (accuracy) vs. the no-information rate:
prop.test(c(116,107),c(170,170))

# You can see that the accuracy is not statistically significantly better than the no-
# information rate.
# You can use the following code to setup 10-fold cross validation:
control <- trainControl(method="repeatedcv", number=10, savePredictions = TRUE)
model <- train(diagnosis~., method="knn", data= trainingData, trControl=control)
model

# Even with 10-fold CV, a model where is k=5 is still the best K-NN classifier.
# You can see the predictions for each fold using the following code:
cv<-model$pred
cv

# You can see that there were 10-folds as expected:
table(model$pred$Resample)

names(getModelInfo())
