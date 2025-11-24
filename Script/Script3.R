#####################################################
# 1. Setup
#####################################################

# install.packages(c("tidyverse", "tidytext", "textdata"))  # run once
library(tidyverse)
library(tidytext)
library(textdata)


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
    text = paste(`review title`, `review content`, sep = ". ")
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
# 4. Sentiment classification (positive/neutral/negative)
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
    sentiment_class = case_when(
      sentiment_score > 0  ~ "positive",
      sentiment_score < 0  ~ "negative",
      TRUE                 ~ "neutral"
    )
  )

# Quick check
count(reviews_sent, sentiment_class)

#####################################################
# 5. Visualise sentiment distribution
#####################################################

ggplot(reviews_sent, aes(x = sentiment_class)) +
  geom_bar() +
  labs(
    title = "Sentiment of Commerce Reviews",
    x = "Sentiment class",
    y = "Number of reviews"
  )

#####################################################
# 6. Word frequency (overall)
#####################################################

word_freq <- tokens %>%
  count(word, sort = TRUE)

top20 <- word_freq %>% slice_head(n = 20)

ggplot(top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 20 Most Frequent Words",
    x = "Word",
    y = "Frequency"
  )

#####################################################
# 7. Word frequency by sentiment class
#####################################################

# Helper functions from tidytext for faceted word plots
# (they’re already in tidytext, just using them)
# reorder_within(word, n, sentiment_class)
# scale_x_reordered()

tokens_with_sent <- tokens %>%
  inner_join(bing_lex, by = "word") %>%
  inner_join(
    reviews_sent %>% select(id, sentiment_class),
    by = "id"
  )

top_words_by_sent <- tokens_with_sent %>%
  count(sentiment_class, word, sort = TRUE) %>%
  group_by(sentiment_class) %>%
  slice_max(n, n = 10) %>%
  ungroup()

ggplot(top_words_by_sent,
       aes(x = reorder_within(word, n, sentiment_class), y = n)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ sentiment_class, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(
    title = "Top Words by Sentiment Class",
    x = "Word",
    y = "Frequency"
  )

#####################################################
# 8. (Optional) Save dataset with sentiment labels
#####################################################

write_csv(reviews_sent, "ebay_reviews_with_sentiment.csv")
