

# Now build a model using superp with caret using the training set:
model <- train(diagnosis~., method="ada", data= trainingData)
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
model <- train(diagnosis~., method="ada", data= trainingData, trControl=control)
model