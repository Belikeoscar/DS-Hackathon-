# Load required libraries
library(tidyverse)
library(tidytext)
library(textdata)
library(sentimentr)
library(wordcloud)
library(tm)
library(SnowballC)
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(stringr)

# Read the dataset
reviews_data <- read.csv("ebay_reviews.csv", stringsAsFactors = FALSE)

# Display basic information about the dataset
cat("Dataset Overview:\n")
cat("Number of reviews:", nrow(reviews_data), "\n")
cat("Number of categories:", length(unique(reviews_data$category)), "\n")
cat("Categories:", paste(unique(reviews_data$category), collapse = ", "), "\n\n")

# Check for missing values
cat("Missing values:\n")
print(colSums(is.na(reviews_data)))

# Data preprocessing
reviews_clean <- reviews_data %>%
  # Remove rows with missing review content
  filter(!is.na(review.content) & review.content != "") %>%
  # Create a combined text field (title + content)
  mutate(
    full_text = paste(ifelse(is.na(review.title), "", review.title), 
                     review.content, sep = ". "),
    full_text = str_squish(full_text),
    # Clean text: remove special characters, convert to lowercase
    clean_text = str_to_lower(full_text),
    clean_text = str_replace_all(clean_text, "[^a-zA-Z0-9\\s]", ""),
    clean_text = str_squish(clean_text)
  ) %>%
  # Remove empty texts after cleaning
  filter(clean_text != "" & !is.na(clean_text))

# Basic sentiment analysis using sentimentr
cat("Performing sentiment analysis...\n")
reviews_with_sentiment <- reviews_clean %>%
  mutate(
    sentiment_score = sentiment_by(clean_text)$ave_sentiment,
    # Classify based on sentiment score
    sentiment_label = case_when(
      sentiment_score > 0.1 ~ "positive",
      sentiment_score < -0.1 ~ "negative",
      TRUE ~ "neutral"
    )
  )

# Display sentiment distribution
sentiment_distribution <- table(reviews_with_sentiment$sentiment_label)
cat("\nSentiment Distribution:\n")
print(sentiment_distribution)
cat("\nSentiment Proportions:\n")
print(prop.table(sentiment_distribution))

# Tokenization and word frequency analysis
cat("\nPerforming tokenization and word frequency analysis...\n")

# Tokenize the text
tokens <- reviews_with_sentiment %>%
  unnest_tokens(word, clean_text) %>%
  # Remove stop words
  anti_join(stop_words, by = "word") %>%
  # Remove numbers and very short words
  filter(!str_detect(word, "^[0-9]+$"),
         nchar(word) > 2)

# Overall word frequency
overall_word_freq <- tokens %>%
  count(word, sort = TRUE) %>%
  top_n(30, n)

# Word frequency by sentiment
sentiment_word_freq <- tokens %>%
  count(sentiment_label, word, sort = TRUE) %>%
  group_by(sentiment_label) %>%
  top_n(15, n) %>%
  ungroup()

# VISUALIZATIONS

# 1. Sentiment distribution pie chart
ggplot(reviews_with_sentiment, aes(x = "", fill = sentiment_label)) +
  geom_bar(width = 1) +
  coord_polar("y") +
  labs(title = "Sentiment Distribution of E-commerce Reviews",
       fill = "Sentiment") +
  theme_void() +
  scale_fill_brewer(palette = "Set2")

# 2. Overall word frequency
ggplot(overall_word_freq, aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 30 Most Frequent Words in Reviews",
       x = "Words",
       y = "Frequency") +
  theme_minimal()

# 3. Word frequency by sentiment
ggplot(sentiment_word_freq, aes(x = reorder(word, n), y = n, fill = sentiment_label)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~sentiment_label, scales = "free") +
  labs(title = "Top Words by Sentiment Category",
       x = "Words",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")

# 4. Word clouds for each sentiment
# Positive reviews word cloud
positive_words <- tokens %>%
  filter(sentiment_label == "positive") %>%
  count(word, sort = TRUE)

negative_words <- tokens %>%
  filter(sentiment_label == "negative") %>%
  count(word, sort = TRUE)

neutral_words <- tokens %>%
  filter(sentiment_label == "neutral") %>%
  count(word, sort = TRUE)

# Create word clouds
par(mfrow = c(1, 3), mar = c(0, 0, 2, 0))

if(nrow(positive_words) > 0) {
  wordcloud(positive_words$word, positive_words$n, 
            max.words = 50, 
            colors = brewer.pal(8, "Dark2"),
            main = "Positive Reviews")
}

if(nrow(negative_words) > 0) {
  wordcloud(negative_words$word, negative_words$n, 
            max.words = 50, 
            colors = brewer.pal(8, "Dark2"),
            main = "Negative Reviews")
}

if(nrow(neutral_words) > 0) {
  wordcloud(neutral_words$word, neutral_words$n, 
            max.words = 50, 
            colors = brewer.pal(8, "Dark2"),
            main = "Neutral Reviews")
}

# Reset plotting parameters
par(mfrow = c(1, 1))

# 5. Sentiment by product category
category_sentiment <- reviews_with_sentiment %>%
  group_by(category, sentiment_label) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(category) %>%
  mutate(percentage = count / sum(count) * 100)

ggplot(category_sentiment, aes(x = category, y = percentage, fill = sentiment_label)) +
  geom_col(position = "fill") +
  coord_flip() +
  labs(title = "Sentiment Distribution by Product Category",
       x = "Category",
       y = "Proportion",
       fill = "Sentiment") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()

# 6. Rating vs Sentiment analysis
rating_sentiment <- reviews_with_sentiment %>%
  group_by(rating, sentiment_label) %>%
  summarise(count = n(), .groups = "drop")

ggplot(rating_sentiment, aes(x = factor(rating), y = count, fill = sentiment_label)) +
  geom_col(position = "dodge") +
  labs(title = "Rating vs Sentiment Analysis",
       x = "Rating",
       y = "Number of Reviews",
       fill = "Sentiment") +
  scale_fill_brewal(palette = "Set2") +
  theme_minimal()

# Advanced Analysis: Bigrams
cat("\nAnalyzing common phrases (bigrams)...\n")

bigrams <- reviews_with_sentiment %>%
  unnest_tokens(bigram, clean_text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "^[0-9]+$"),
         !str_detect(word2, "^[0-9]+$"),
         nchar(word1) > 2,
         nchar(word2) > 2) %>%
  unite(bigram, word1, word2, sep = " ")

# Top bigrams by sentiment
bigram_freq <- bigrams %>%
  count(sentiment_label, bigram, sort = TRUE) %>%
  group_by(sentiment_label) %>%
  top_n(10, n) %>%
  ungroup()

# Plot top bigrams
ggplot(bigram_freq, aes(x = reorder(bigram, n), y = n, fill = sentiment_label)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~sentiment_label, scales = "free") +
  labs(title = "Top Bigrams by Sentiment Category",
       x = "Bigrams",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")

# Export results
cat("\nExporting results...\n")

# Save processed data with sentiment labels
write.csv(reviews_with_sentiment, "reviews_with_sentiment_analysis.csv", row.names = FALSE)

# Save word frequency tables
write.csv(overall_word_freq, "overall_word_frequency.csv", row.names = FALSE)
write.csv(sentiment_word_freq, "sentiment_word_frequency.csv", row.names = FALSE)

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
cat("Total reviews analyzed:", nrow(reviews_with_sentiment), "\n")
cat("Positive reviews:", sum(reviews_with_sentiment$sentiment_label == "positive"), 
    "(", round(mean(reviews_with_sentiment$sentiment_label == "positive") * 100, 1), "%)\n")
cat("Neutral reviews:", sum(reviews_with_sentiment$sentiment_label == "neutral"), 
    "(", round(mean(reviews_with_sentiment$sentiment_label == "neutral") * 100, 1), "%)\n")
cat("Negative reviews:", sum(reviews_with_sentiment$sentiment_label == "negative"), 
    "(", round(mean(reviews_with_sentiment$sentiment_label == "negative") * 100, 1), "%)\n")

# Average sentiment by category
cat("\nAverage Sentiment Score by Category:\n")
category_avg_sentiment <- reviews_with_sentiment %>%
  group_by(category) %>%
  summarise(avg_sentiment = mean(sentiment_score),
            n_reviews = n()) %>%
  arrange(desc(avg_sentiment))

print(category_avg_sentiment)

cat("\nAnalysis complete! Check the generated visualizations and CSV files.\n")
