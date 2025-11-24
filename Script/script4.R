#####################################################
# 1. Setup
#####################################################

# install.packages(c("tidyverse", "tidytext", "textdata", "randomForest", "tm", "caret"))  # run once
library(tidyverse)
library(tidytext)
library(textdata)
library(randomForest)
library(tm)
library(caret)
library(ggpubr)  # for arranging multiple plots

#####################################################
# 2. Load data
#####################################################

setwd("C:/Users/user/Documents/GitHub/Data_Science")

# Make sure working directory contains ebay_reviews.csv
reviews_raw <- read_csv("ebay_reviews.csv")

glimpse(reviews_raw)

# Create an ID and combine title + content into one text column
reviews <- reviews_raw %>%
  mutate(
    id   = row_number(),
    text = paste(`review title`, `review content`, sep = ". "),
    # Create a balanced dataset for ML (limit to 500 of each class for demo)
    rating_class = case_when(
      `review rating` >= 4 ~ "positive",
      `review rating` == 3 ~ "neutral", 
      `review rating` <= 2 ~ "negative"
    )
  )

#####################################################
# 3. Tokenise text & clean
#####################################################

data("stop_words")

tokens <- reviews %>%
  select(id, text) %>%
  unnest_tokens(word, text) %>%         # one word per row
  anti_join(stop_words, by = "word") %>%# remove stopwords
  filter(!str_detect(word, "^[0-9]+$")) # drop pure numbers

#####################################################
# 4. Sentiment classification (BASELINE - Lexicon-based)
#####################################################

# Bing lexicon: each word is labelled positive or negative
bing_lex <- get_sentiments("bing")

# Join words with sentiment and compute score per review
review_scores <- tokens %>%
  inner_join(bing_lex, by = "word") %>%
  mutate(score = if_else(sentiment == "positive", 1, -1)) %>%
  group_by(id) %>%
  summarise(sentiment_score = sum(score), .groups = "drop")

# Attach score back to main data and classify
reviews_sent <- reviews %>%
  left_join(review_scores, by = "id") %>%
  mutate(
    sentiment_score = replace_na(sentiment_score, 0),
    sentiment_class_baseline = case_when(
      sentiment_score > 0  ~ "positive",
      sentiment_score < 0  ~ "negative",
      TRUE                 ~ "neutral"
    )
  )

#####################################################
# 5. MACHINE LEARNING: Random Forest Preparation
#####################################################

# Create Document-Term Matrix for ML
create_dtm <- function(text_data) {
  corpus <- Corpus(VectorSource(text_data))
  
  # Preprocessing
  corpus <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords("english")) %>%
    tm_map(stripWhitespace)
  
  # Create DTM and remove sparse terms
  dtm <- DocumentTermMatrix(corpus)
  dtm <- removeSparseTerms(dtm, 0.95)  # Keep terms that appear in at least 5% of documents
  
  return(dtm)
}

# Prepare data for ML
set.seed(123)  # for reproducibility

# Balance the dataset (for demonstration)
ml_data <- reviews %>%
  filter(!is.na(rating_class)) %>%
  group_by(rating_class) %>%
  slice_sample(n = min(300, n())) %>%  # Take 300 from each class or minimum available
  ungroup()

# Create DTM
dtm <- create_dtm(ml_data$text)

# Convert to dataframe and add target variable
ml_df <- as.data.frame(as.matrix(dtm))
ml_df$sentiment_class <- as.factor(ml_data$rating_class)

# Split data (80% train, 20% test)
train_idx <- createDataPartition(ml_df$sentiment_class, p = 0.8, list = FALSE)
train_data <- ml_df[train_idx, ]
test_data <- ml_df[-train_idx, ]

#####################################################
# 6. Train Random Forest Model
#####################################################

cat("Training Random Forest model...\n")

# Train Random Forest
rf_model <- randomForest(
  sentiment_class ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE
)

# Make predictions
rf_predictions <- predict(rf_model, test_data)

# Evaluate model
conf_matrix <- confusionMatrix(rf_predictions, test_data$sentiment_class)
print(conf_matrix)

#####################################################
# 7. Apply RF Model to Full Dataset & Compare
#####################################################

# Apply RF to full dataset (using the same features)
full_dtm <- create_dtm(reviews$text)

# Align features with training data
full_df <- as.data.frame(as.matrix(full_dtm))
common_features <- intersect(names(full_df), names(train_data))

# Ensure all training features exist in full dataset
missing_features <- setdiff(names(train_data), names(full_df))
for(feature in missing_features) {
  if(feature != "sentiment_class") {
    full_df[[feature]] <- 0
  }
}

# Keep only the features used in training
full_df <- full_df[, names(train_data)[names(train_data) != "sentiment_class"]]

# Make predictions on full dataset
full_rf_predictions <- predict(rf_model, full_df)

# Add RF predictions to main dataset
reviews_sent$sentiment_class_rf <- full_rf_predictions

#####################################################
# 8. Comparative Visualizations
#####################################################

# Create comparison dataframe
comparison_df <- reviews_sent %>%
  select(sentiment_class_baseline, sentiment_class_rf) %>%
  gather(method, sentiment, sentiment_class_baseline:sentiment_class_rf) %>%
  mutate(method = case_when(
    method == "sentiment_class_baseline" ~ "Baseline (Lexicon)",
    method == "sentiment_class_rf" ~ "Random Forest"
  ))

# Plot 1: Side-by-side comparison
p1 <- ggplot(comparison_df, aes(x = sentiment, fill = method)) +
  geom_bar(position = "dodge") +
  facet_wrap(~ method, ncol = 2) +
  labs(
    title = "Sentiment Distribution: Baseline vs Random Forest",
    x = "Sentiment Class",
    y = "Number of Reviews",
    fill = "Method"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

# Plot 2: RF-only distribution
p2 <- reviews_sent %>%
  count(sentiment_class_rf) %>%
  ggplot(aes(x = sentiment_class_rf, y = n, fill = sentiment_class_rf)) +
  geom_col() +
  labs(
    title = "Random Forest Sentiment Distribution",
    x = "Sentiment Class",
    y = "Number of Reviews",
    fill = "Sentiment"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("negative" = "red", "neutral" = "gray", "positive" = "green"))

# Plot 3: Comparison as stacked percentage
p3 <- comparison_df %>%
  count(method, sentiment) %>%
  group_by(method) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  ggplot(aes(x = method, y = percentage, fill = sentiment)) +
  geom_col(position = "stack") +
  labs(
    title = "Sentiment Proportion Comparison",
    x = "Method",
    y = "Percentage (%)",
    fill = "Sentiment"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("negative" = "red", "neutral" = "gray", "positive" = "green"))

# Plot 4: Feature importance from Random Forest
importance_df <- as.data.frame(rf_model$importance) %>%
  rownames_to_column("word") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(20)

p4 <- ggplot(importance_df, aes(x = reorder(word, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Random Forest - Top 20 Important Features",
    x = "Word",
    y = "Importance (Mean Decrease Gini)"
  ) +
  theme_minimal()

# Display all plots
print(p1)
print(p2)
print(p3)
print(p4)

# Arrange in grid (if you want them together)
grid_plot <- ggarrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
print(grid_plot)

#####################################################
# 9. Save ML Results
#####################################################

# Save only Random Forest results
rf_results <- reviews_sent %>%
  select(-sentiment_class_baseline, -sentiment_score) %>%
  rename(sentiment_class = sentiment_class_rf)

write_csv(rf_results, "ebay_reviews_rf_sentiment.csv")

cat("Random Forest results saved to 'ebay_reviews_rf_sentiment.csv'\n")

#####################################################
# 10. Model Performance Summary
#####################################################

cat("\n=== MODEL PERFORMANCE SUMMARY ===\n")
cat("Random Forest Accuracy on Test Set:", round(conf_matrix$overall['Accuracy'] * 100, 2), "%\n")

# Baseline accuracy (if we had true labels)
baseline_dist <- count(reviews_sent, sentiment_class_baseline) %>%
  mutate(percentage = n / sum(n) * 100)

rf_dist <- count(reviews_sent, sentiment_class_rf) %>%
  mutate(percentage = n / sum(n) * 100)

cat("\nBaseline Distribution:\n")
print(baseline_dist)

cat("\nRandom Forest Distribution:\n")
print(rf_dist)

cat("\nComparison complete! Check the visualizations for insights.\n")
