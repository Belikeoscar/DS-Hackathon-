#####################################################
# 1. Setup
#####################################################

# Run once if needed:
# install.packages(c("tidyverse", "tidytext", "randomForest", "tm", "caret"))

library(tidyverse)
library(tidytext)
library(randomForest)
library(tm)
library(caret)

set.seed(123)

#####################################################
# 2. Load data
#####################################################

setwd("C:/Users/user/Documents/GitHub/Data_Science")

reviews_raw <- read_csv("ebay_reviews.csv")

glimpse(reviews_raw)

# Create ID, combined text, and ML target (sentiment from rating)
reviews <- reviews_raw %>%
  mutate(
    id   = row_number(),
    text = paste(`review title`, `review content`, sep = ". "),
    rating_class = case_when(
      rating >= 4 ~ "positive",
      rating == 3 ~ "neutral",
      rating <= 2 ~ "negative"
    ),
    rating_class = factor(rating_class,
                          levels = c("negative", "neutral", "positive"))
  )

#####################################################
# 3. Build Document-Term Matrix (ALL reviews)
#####################################################

corpus <- VCorpus(VectorSource(reviews$text))

# Pre-process text
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Create DTM and remove very sparse terms
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.995)  # tweak if needed

# Convert to data frame
dtm_df <- as.data.frame(as.matrix(dtm))

# ⬇️ FIX: add id and rating_class using base R (no mutate)
dtm_df$id           <- reviews$id
dtm_df$rating_class <- reviews$rating_class

#####################################################
# 4. Prepare data for Random Forest (balanced sample)
#####################################################

ml_df <- dtm_df %>%
  filter(!is.na(rating_class))

# Balance: up to 300 docs per class
ml_bal <- ml_df %>%
  group_by(rating_class) %>%
  group_modify(~ {
    size <- min(300, nrow(.x))
    dplyr::slice_sample(.x, n = size)
  }) %>%
  ungroup()

# Train/Test split
train_idx  <- createDataPartition(ml_bal$rating_class, p = 0.8, list = FALSE)
train_data <- ml_bal[train_idx, ]
test_data  <- ml_bal[-train_idx, ]

train_x <- train_data %>% select(-id, -rating_class)
train_y <- train_data$rating_class

test_x  <- test_data %>% select(-id, -rating_class)
test_y  <- test_data$rating_class

#####################################################
# 5. Train Random Forest model
#####################################################

cat("Training Random Forest model...\n")

rf_model <- randomForest(
  x         = train_x,
  y         = train_y,
  ntree     = 200,
  importance = TRUE
)

rf_preds_test <- predict(rf_model, newdata = test_x)
conf_matrix   <- confusionMatrix(rf_preds_test, test_y)
print(conf_matrix)

#####################################################
# 6. Predict sentiment for ALL reviews
#####################################################

all_x   <- ml_df %>% select(-id, -rating_class)
all_ids <- ml_df$id

all_preds <- predict(rf_model, newdata = all_x)

reviews$sentiment_rf <- NA_character_
reviews$sentiment_rf[match(all_ids, reviews$id)] <- as.character(all_preds)

#####################################################
# 7. PLOT 1 – Sentiment distribution (RF predictions)
#####################################################

sentiment_counts <- reviews %>%
  filter(!is.na(sentiment_rf)) %>%
  count(sentiment_rf) %>%
  mutate(
    percentage = n / sum(n) * 100,
    label = paste0(sentiment_rf, "\n", round(percentage, 1), "%")
  )

ggplot(sentiment_counts,
       aes(x = "", y = n, fill = sentiment_rf)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            color = "white", fontface = "bold", size = 5) +
  scale_fill_manual(values = c(
    "positive" = "#27AE60",
    "neutral"  = "#F39C12",
    "negative" = "#E74C3C"
  )) +
  labs(
    title    = "Sentiment Distribution – Random Forest Predictions",
    subtitle = paste("Total Reviews:", sum(sentiment_counts$n)),
    fill     = "Sentiment"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    legend.position = "none"
  )

#####################################################
# 8. PLOT 2 – Top 15 words by RF sentiment
#####################################################

data("stop_words")

tokens <- reviews %>%
  select(id, text) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(!str_detect(word, "^[0-9]+$"))

tokens_pred <- tokens %>%
  inner_join(
    reviews %>% select(id, sentiment_rf),
    by = "id"
  ) %>%
  filter(!is.na(sentiment_rf))

sentiment_words <- tokens_pred %>%
  count(sentiment_rf, word, sort = TRUE) %>%
  group_by(sentiment_rf) %>%
  slice_max(n, n = 15) %>%
  ungroup()

ggplot(sentiment_words,
       aes(x = reorder_within(word, n, sentiment_rf),
           y = n,
           fill = sentiment_rf)) +
  geom_col(show.legend = FALSE) +
  scale_x_reordered() +
  facet_wrap(~ sentiment_rf, scales = "free_y", ncol = 1) +
  coord_flip() +
  scale_fill_manual(values = c(
    "positive" = "#27AE60",
    "neutral"  = "#F39C12",
    "negative" = "#E74C3C"
  )) +
  labs(
    title    = "Top 15 Words by Sentiment – Random Forest",
    subtitle = "Most frequent words in positive, neutral, and negative reviews",
    x        = "Words",
    y        = "Frequency"
  ) +
  theme_minimal()

#####################################################
# 9. Save results (optional)
#####################################################

rf_results <- reviews %>%
  select(category, `review title`, `review content`, rating, sentiment_rf)

write_csv(rf_results, "ebay_reviews_rf_sentiment.csv")

cat("\nRandom Forest results saved to 'ebay_reviews_rf_sentiment.csv'\n")
cat("Test accuracy:", round(conf_matrix$overall['Accuracy'] * 100, 2), "%\n")
