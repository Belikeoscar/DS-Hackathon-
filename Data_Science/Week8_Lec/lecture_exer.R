mtcars
mtcars <- mtcars

mtcarsNew <- mtcars[,c(1,2,4)]

mtcarsNew <- scale(mtcarsNew)

set.seed(123)

kMeansClust <- kmeans(mtcarsNew, centers = 3, iter.max = 10, nstart = 25)

kMeansClust$cluster

kMeansClust$centers

mtcars$clusterLabel <- kMeansClust$cluster

boxplot(mtcars$mpg ~ mtcars$clusterLabel)
