# Enhanced script with ML models and comparison
library(tidyverse)
library(tidytext)
library(textdata)
library(sentimentr)
library(wordcloud)
library(tm)
library(SnowballC)
library(caret)
library(randomForest)
library(glmnet)
library(text2vec)
library(xgboost)
library(yardstick)

# Read and preprocess data (same as before)
reviews_data <- read.csv("ebay_reviews.csv", stringsAsFactors = FALSE)
reviews_clean <- reviews_data %>%
  filter(!is.na(review.content) & review.content != "") %>%
  mutate(
    full_text = paste(ifelse(is.na(review.title), "", review.title), 
                     review.content, sep = ". "),
    full_text = str_squish(full_text),
    clean_text = str_to_lower(full_text),
    clean_text = str_replace_all(clean_text, "[^a-zA-Z0-9\\s]", ""),
    clean_text = str_squish(clean_text)
  ) %>%
  filter(clean_text != "" & !is.na(clean_text))

# Apply sentimentr for baseline
reviews_with_sentiment <- reviews_clean %>%
  mutate(
    sentiment_score = sentiment_by(clean_text)$ave_sentiment,
    sentiment_label = case_when(
      sentiment_score > 0.1 ~ "positive",
      sentiment_score < -0.1 ~ "negative",
      TRUE ~ "neutral"
    )
  )

# MACHINE LEARNING IMPLEMENTATION
prepare_ml_features <- function(texts) {
  # Create document-term matrix with TF-IDF
  it <- itoken(texts, 
               preprocessor = tolower,
               tokenizer = word_tokenizer,
               ids = seq_along(texts),
               progressbar = FALSE)
  
  # Create vocabulary and prune
  vocab <- create_vocabulary(it)
  vocab <- prune_vocabulary(vocab, term_count_min = 5)
  
  # Create vectorizer and DTM
  vectorizer <- vocab_vectorizer(vocab)
  dtm <- create_dtm(it, vectorizer)
  
  # Apply TF-IDF transformation
  tfidf <- TfIdf$new()
  dtm_tfidf <- fit_transform(dtm, tfidf)
  
  return(as.matrix(dtm_tfidf))
}

# Prepare ML dataset
cat("Preparing data for machine learning...\n")
ml_texts <- reviews_with_sentiment$clean_text

# Create labels from ratings (since we don't have true sentiment labels)
# Using rating as proxy: 4-5 = positive, 3 = neutral, 1-2 = negative
ml_labels <- factor(case_when(
  reviews_with_sentiment$rating >= 4 ~ "positive",
  reviews_with_sentiment$rating <= 2 ~ "negative",
  TRUE ~ "neutral"
), levels = c("negative", "neutral", "positive"))

# Create features
features <- prepare_ml_features(ml_texts)

# Split data
set.seed(123)
train_index <- createDataPartition(ml_labels, p = 0.8, list = FALSE)
train_features <- features[train_index, ]
test_features <- features[-train_index, ]
train_labels <- ml_labels[train_index]
test_labels <- ml_labels[-train_index]

# MODEL 1: Random Forest
cat("Training Random Forest...\n")
rf_model <- randomForest(
  x = train_features,
  y = train_labels,
  ntree = 100,
  importance = TRUE,
  do.trace = FALSE
)

rf_predictions <- predict(rf_model, test_features)
rf_accuracy <- mean(rf_predictions == test_labels)

# MODEL 2: XGBoost
cat("Training XGBoost...\n")
xgb_model <- xgboost(
  data = train_features,
  label = as.numeric(train_labels) - 1,  # XGBoost expects 0-based classes
  nrounds = 100,
  objective = "multi:softmax",
  num_class = 3,
  eval_metric = "mlogloss",
  verbose = 0
)

xgb_predictions <- factor(levels(ml_labels)[predict(xgb_model, test_features) + 1],
                         levels = levels(ml_labels))
xgb_accuracy <- mean(xgb_predictions == test_labels)

# Baseline: sentimentr
sentimentr_predictions <- reviews_with_sentiment$sentiment_label[-train_index]
sentimentr_accuracy <- mean(sentimentr_predictions == test_labels)

# COMPREHENSIVE COMPARISON
cat("\n=== MODEL COMPARISON ===\n")
comparison_results <- data.frame(
  Model = c("sentimentr (Baseline)", "Random Forest", "XGBoost"),
  Accuracy = c(sentimentr_accuracy, rf_accuracy, xgb_accuracy),
  Precision = c(
    precision_vec(factor(sentimentr_predictions), test_labels),
    precision_vec(rf_predictions, test_labels),
    precision_vec(xgb_predictions, test_labels)
  ),
  Recall = c(
    recall_vec(factor(sentimentr_predictions), test_labels),
    recall_vec(rf_predictions, test_labels),
    recall_vec(xgb_predictions, test_labels)
  ),
  F1_Score = c(
    f_meas_vec(factor(sentimentr_predictions), test_labels),
    f_meas_vec(rf_predictions, test_labels),
    f_meas_vec(xgb_predictions, test_labels)
  )
)

print(comparison_results)

# Select best model
best_model_name <- comparison_results$Model[which.max(comparison_results$Accuracy)]
cat("\nBest performing model:", best_model_name, "\n")

# Use best model for final predictions
if (best_model_name == "Random Forest") {
  final_predictions <- predict(rf_model, features)
  best_model <- rf_model
} else if (best_model_name == "XGBoost") {
  final_predictions <- factor(levels(ml_labels)[predict(xgb_model, features) + 1],
                             levels = levels(ml_labels))
  best_model <- xgb_model
} else {
  final_predictions <- factor(reviews_with_sentiment$sentiment_label,
                             levels = levels(ml_labels))
  best_model <- "sentimentr"
}

# Create final dataset with ML predictions
final_dataset <- reviews_with_sentiment %>%
  mutate(
    ml_sentiment_label = final_predictions,
    ml_confidence = if(best_model_name != "sentimentr") {
      # For ML models, get prediction probabilities
      if (best_model_name == "Random Forest") {
        apply(predict(rf_model, features, type = "prob"), 1, max)
      } else {
        apply(predict(xgb_model, features, type = "prob"), 1, max)
      }
    } else {
      NA_real_
    },
    final_sentiment = ml_sentiment_label,  # Use best model predictions
    analysis_method = best_model_name
  )

# Export final CSV
final_output <- final_dataset %>%
  select(category, review.title, review.content, rating, 
         sentiment_score, sentiment_label, ml_sentiment_label, final_sentiment,
         ml_confidence, analysis_method)

write.csv(final_output, "final_review_analysis_with_ml.csv", row.names = FALSE)
cat("Final CSV exported: 'final_review_analysis_with_ml.csv'\n")

# Generate comparison visualizations
comparison_long <- comparison_results %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col() +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Machine Learning Model Comparison",
       subtitle = "Performance across different metrics",
       y = "Score", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Confusion matrices
create_confusion_plot <- function(predictions, true_labels, model_name) {
  cm <- table(Predicted = predictions, Actual = true_labels)
  cm_df <- as.data.frame(cm)
  
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", fontface = "bold") +
    scale_fill_gradient(low = "blue", high = "red") +
    labs(title = paste("Confusion Matrix -", model_name),
         x = "Actual Sentiment", y = "Predicted Sentiment") +
    theme_minimal()
}

# Create confusion matrices for all models
p1 <- create_confusion_plot(factor(sentimentr_predictions), test_labels, "sentimentr")
p2 <- create_confusion_plot(rf_predictions, test_labels, "Random Forest") 
p3 <- create_confusion_plot(xgb_predictions, test_labels, "XGBoost")

# Print summary
cat("\n=== FINAL SUMMARY ===\n")
cat("Best Model:", best_model_name, "\n")
cat("Best Accuracy:", round(max(comparison_results$Accuracy) * 100, 2), "%\n")
cat("Total Reviews Analyzed:", nrow(final_output), "\n")
cat("Final File: 'final_review_analysis_with_ml.csv'\n")

# Show feature importance for best ML model
if (best_model_name == "Random Forest") {
  cat("\nTop 10 Important Features (Random Forest):\n")
  imp <- importance(rf_model)
  print(head(imp[order(-imp[, "MeanDecreaseAccuracy"]), ], 10))
}
