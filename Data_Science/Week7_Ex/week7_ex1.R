library(janeaustenr)
?word2vec

# The following code should allow you to build word vectors/word embeddings
# using this code. Here we have only selected one book to build word vectors. And
# we have chosen to represent words using a vector of 15 dimensions (15 numbers).
# We have also selected to use ‘cbow’ (continuous bag of words) to build the model.

set.seed(1)
txt <- austen_books()
model <- word2vec(x = txt$text[txt$book == "Sense & Sensibility"], type = "cbow", dim = 15, iter = 20)
embeddings <- as.matrix(model)

# This code should allow you to see the word vectors:
View(embeddings)

# Have a look at the so called vocabulary in this model and the number of word
# embeddings (~2141 words):
vocabulary <- summary(model, type = "vocabulary")
vocabulary
length(vocabulary)
# Note: This is important since it is determined by the text you used to train the model.
# Hence, words ‘out of vocabulary’ will not have a word vector.

# You can also select a word vector using the following code:
embeddings["hope",]

# You should now be able to study neighbouring words according to the vector space:
neighbourWords <- predict(model, c("Lucy"), type = "nearest", top_n = 50)
neighbourWords

# Now try typing in more words:
neighbourWords <- predict(model, c("Lucy", "man", "woman"), type = "nearest", top_n = 10)
neighbourWords

# Develop a word2vec model using just 2 dimensions
set.seed(1)
txt <- austen_books()
model <- word2vec(x = txt$text[txt$book == "Sense & Sensibility"], type = "cbow", dim = 2, iter = 20)
embeddings <- as.matrix(model)
embeddings <- predict(model, c(
  "John","Lucy","Edward","Marianne","Elinor","woman","man","she",
  "he","Colonel","Brandon", "rain", "weather"), type = "embedding")

# Given these words are now represented using just 2 values (2 dimensions), we
# should now be able to visualise this, example:
embeddings <- as.data.frame(embeddings)
plot(embeddings$V1, embeddings$V2)
text(embeddings$V1, embeddings$V2, labels = row.names(embeddings), pos = 1)

# Try calculating the Euclidean distance between word vectors:
# Source: https://www.statology.org/euclidean-distance-in-r/
euclidean <- function(a, b) sqrt(sum((a - b)^2))
euclidean(embeddings["rain",], embeddings["weather",])
euclidean(embeddings["Brandon",], embeddings["Colonel",])

# You should also be able to calculate the cosine similarity using the following:
# Source: https://www.r-bloggers.com/2021/08/how-to-calculate-cosine-similarity-in-r/
library(lsa)
set.seed(1)
txt <- austen_books()
model <- word2vec(x = txt$text[txt$book == "Sense & Sensibility"], type = "cbow", dim = 2, iter = 20)
embeddings <- as.matrix(model)
cosine(embeddings["Lucy",], embeddings["Elinor",])

# Try calculating the correlation between different word vectors:
set.seed(1)
txt <- austen_books()
model <- word2vec(x = txt$text[txt$book == "Sense & Sensibility"], type = "cbow", dim = 15, iter = 20)
embeddings <- as.matrix(model)
cor(embeddings["Brandon",], embeddings["Colonel",])
cor(embeddings["weather",], embeddings["rain",])

# In theory, you should be able to do arithmetic with word vectors, consider the
# following as an example:
set.seed(1)
txt <- austen_books()
model <- word2vec(x = txt$text[txt$book == "Sense & Sensibility"], type = "cbow", dim = 2, iter = 20)
embeddings <- as.matrix(model)
vector <- embeddings["hurry", ] + embeddings["walk", ]
predict(model, vector, type = "nearest", top_n = 10)
