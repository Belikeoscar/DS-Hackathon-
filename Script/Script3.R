install.packages("ggwordcloud")

# Load required libraries
library(readr)
library(dplyr)
library(stringr)
library(tidytext)
library(sentimentr)
library(randomForest)
library(caret)
library(ggplot2)
library(wordcloud)
library(tm)
library(SnowballC)
library(scales)
library(ggwordcloud)

# Read the data
reviews <- read_csv("ebay_reviews.csv")

# Data preprocessing
clean_text <- function(text) {
  text %>%
    str_to_lower() %>%
    str_replace_all("[^a-zA-Z\\s]", " ") %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

reviews_clean <- reviews %>%
  mutate(
    clean_review = clean_text("review content"),
    # Create sentiment labels from ratings
    sentiment_label = case_when(
      rating >= 4 ~ "positive",
      rating == 3 ~ "neutral",
      rating <= 2 ~ "negative"
    )
  ) %>%
  filter(!is.na(clean_review) & clean_review != "")

# 1. Overall Most Used Words Visualization
create_overall_word_frequency <- function(data) {
  # Tokenize and count all words
  overall_words <- data %>%
    unnest_tokens(word, clean_review) %>%
    anti_join(stop_words) %>%
    count(word, sort = TRUE) %>%
    head(20)
  
  # Create bar chart
  p1 <- ggplot(overall_words, aes(x = reorder(word, n), y = n)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 20 Most Frequently Used Words",
         subtitle = "Across All Reviews",
         x = "Words",
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14),
          axis.text = element_text(size = 10))
  
  # Create word cloud
  p2 <- ggplot(overall_words, aes(label = word, size = n, color = n)) +
    geom_text_wordcloud_area(rm_outside = TRUE) +
    scale_size_area(max_size = 15) +
    scale_color_gradient(low = "blue", high = "red") +
    labs(title = "Word Cloud - Most Frequent Words") +
    theme_minimal()
  
  return(list(bar_chart = p1, word_cloud = p2))
}

# 2. Most Used Words by Sentiment Category
create_sentiment_word_frequency <- function(data) {
  # Tokenize and count words by sentiment
  sentiment_words <- data %>%
    unnest_tokens(word, clean_review) %>%
    anti_join(stop_words) %>%
    count(sentiment_label, word, sort = TRUE)
  
  # Get top 15 words for each sentiment
  top_sentiment_words <- sentiment_words %>%
    group_by(sentiment_label) %>%
    slice_max(n, n = 15) %>%
    ungroup() %>%
    mutate(word = reorder_within(word, n, sentiment_label))
  
  # Create faceted bar chart
  p1 <- ggplot(top_sentiment_words, aes(word, n, fill = sentiment_label)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~sentiment_label, scales = "free_y") +
    coord_flip() +
    scale_x_reordered() +
    labs(title = "Top 15 Words by Sentiment Category",
         subtitle = "Most Frequently Used Words in Positive, Neutral, and Negative Reviews",
         x = NULL,
         y = "Frequency") +
    theme_minimal() +
    theme(strip.text = element_text(face = "bold", size = 12),
          plot.title = element_text(face = "bold", size = 14))
  
  # Create individual word clouds for each sentiment
  sentiment_list <- list()
  sentiments <- unique(data$sentiment_label)
  
  for(sent in sentiments) {
    words_df <- sentiment_words %>%
      filter(sentiment_label == sent) %>%
      head(30)
    
    p <- ggplot(words_df, aes(label = word, size = n, color = n)) +
      geom_text_wordcloud_area(rm_outside = TRUE) +
      scale_size_area(max_size = 12) +
      scale_color_gradient(low = "darkblue", high = "red") +
      labs(title = paste("Word Cloud -", str_to_title(sent), "Reviews")) +
      theme_minimal()
    
    sentiment_list[[sent]] <- p
  }
  
  return(list(bar_chart = p1, word_clouds = sentiment_list))
}

# 3. Most Used Words by Product Category
create_category_word_frequency <- function(data) {
  # Get top categories
  top_categories <- data %>%
    count(category, sort = TRUE) %>%
    head(6) %>%  # Top 6 categories for visualization
    pull(category)
  
  # Filter data for top categories and get word frequencies
  category_words <- data %>%
    filter(category %in% top_categories) %>%
    unnest_tokens(word, clean_review) %>%
    anti_join(stop_words) %>%
    count(category, word, sort = TRUE)
  
  # Get top 10 words for each category
  top_category_words <- category_words %>%
    group_by(category) %>%
    slice_max(n, n = 10) %>%
    ungroup() %>%
    mutate(word = reorder_within(word, n, category))
  
  # Create faceted plot
  p <- ggplot(top_category_words, aes(word, n, fill = category)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~category, scales = "free", ncol = 2) +
    coord_flip() +
    scale_x_reordered() +
    labs(title = "Top 10 Words by Product Category",
         subtitle = "Most Frequently Used Words in Different Product Categories",
         x = NULL,
         y = "Frequency") +
    theme_minimal() +
    theme(strip.text = element_text(face = "bold", size = 10),
          plot.title = element_text(face = "bold", size = 14),
          axis.text = element_text(size = 8))
  
  return(p)
}

# 4. Word Frequency Comparison: Positive vs Negative
create_sentiment_comparison <- function(data) {
  # Calculate word frequencies and sentiment proportions
  sentiment_ratio <- data %>%
    unnest_tokens(word, clean_review) %>%
    anti_join(stop_words) %>%
    count(word, sentiment_label) %>%
    group_by(word) %>%
    mutate(total_n = sum(n)) %>%
    ungroup() %>%
    filter(total_n > 10) %>%  # Only words that appear at least 10 times
    pivot_wider(names_from = sentiment_label, values_from = n, values_fill = 0) %>%
    mutate(
      positive_ratio = positive / (positive + neutral + negative),
      negative_ratio = negative / (positive + neutral + negative),
      sentiment_difference = positive_ratio - negative_ratio
    ) %>%
    arrange(desc(total_n))
  
  # Get top 20 most frequent words for comparison
  top_comparison_words <- sentiment_ratio %>%
    head(20) %>%
    pivot_longer(cols = c(positive, neutral, negative), 
                 names_to = "sentiment", values_to = "count")
  
  # Create comparison plot
  p <- ggplot(top_comparison_words, aes(x = reorder(word, total_n), y = count, fill = sentiment)) +
    geom_col(position = "fill") +
    coord_flip() +
    scale_fill_manual(values = c("positive" = "#2E8B57", 
                                 "neutral" = "#FFD700", 
                                 "negative" = "#DC143C")) +
    scale_y_continuous(labels = percent) +
    labs(title = "Sentiment Distribution for Top 20 Most Frequent Words",
         subtitle = "Proportion of Positive, Neutral, and Negative Usage",
         x = "Words",
         y = "Proportion",
         fill = "Sentiment") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 14))
  
  return(p)
}

# Generate all visualizations
cat("Generating word frequency visualizations...\n")

# 1. Overall most used words
cat("1. Creating overall word frequency visualizations...\n")
overall_viz <- create_overall_word_frequency(reviews_clean)
print(overall_viz$bar_chart)
print(overall_viz$word_cloud)

# 2. Words by sentiment
cat("2. Creating sentiment-based word frequency visualizations...\n")
sentiment_viz <- create_sentiment_word_frequency(reviews_clean)
print(sentiment_viz$bar_chart)

# Print individual sentiment word clouds
for(sent in names(sentiment_viz$word_clouds)) {
  print(sentiment_viz$word_clouds[[sent]])
}

# 3. Words by category (if enough categories exist)
if(length(unique(reviews_clean$category)) >= 3) {
  cat("3. Creating category-based word frequency visualizations...\n")
  category_viz <- create_category_word_frequency(reviews_clean)
  print(category_viz)
}

# 4. Sentiment comparison for frequent words
cat("4. Creating sentiment comparison visualization...\n")
comparison_viz <- create_sentiment_comparison(reviews_clean)
print(comparison_viz)

# 5. Additional: Bigram analysis for common phrases
create_bigram_analysis <- function(data) {
  # Extract bigrams
  bigrams <- data %>%
    unnest_tokens(bigram, clean_review, token = "ngrams", n = 2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word,
           !word2 %in% stop_words$word) %>%
    unite(bigram, word1, word2, sep = " ") %>%
    count(bigram, sort = TRUE) %>%
    head(15)
  
  # Create bigram plot
  p <- ggplot(bigrams, aes(x = reorder(bigram, n), y = n)) +
    geom_col(fill = "purple", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 15 Most Frequent Bigrams",
         subtitle = "Common Two-Word Phrases in Reviews",
         x = "Bigrams",
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14))
  
  return(p)
}

# Generate bigram analysis
cat("5. Creating bigram analysis...\n")
bigram_viz <- create_bigram_analysis(reviews_clean)
print(bigram_viz)

# Save all visualizations to files
cat("Saving visualizations to files...\n")
ggsave("overall_word_frequency.png", overall_viz$bar_chart, width = 10, height = 8)
ggsave("sentiment_word_frequency.png", sentiment_viz$bar_chart, width = 12, height = 8)

if(length(unique(reviews_clean$category)) >= 3) {
  ggsave("category_word_frequency.png", category_viz, width = 12, height = 10)
}

ggsave("sentiment_comparison.png", comparison_viz, width = 10, height = 8)
ggsave("bigram_analysis.png", bigram_viz, width = 10, height = 8)

# Create summary statistics
cat("\n=== WORD FREQUENCY SUMMARY ===\n")
total_words <- reviews_clean %>%
  unnest_tokens(word, clean_review) %>%
  nrow()

unique_words <- reviews_clean %>%
  unnest_tokens(word, clean_review) %>%
  distinct(word) %>%
  nrow()

cat("Total words in corpus:", total_words, "\n")
cat("Unique words:", unique_words, "\n")
cat("Average words per review:", round(total_words / nrow(reviews_clean), 1), "\n")

# Show top 10 words
top_10_words <- reviews_clean %>%
  unnest_tokens(word, clean_review) %>%
  anti_join(stop_words) %>%
  count(word, sort = TRUE) %>%
  head(10)

cat("\nTop 10 Most Frequent Words:\n")
print(top_10_words)

# Sentiment word summary
sentiment_summary <- reviews_clean %>%
  unnest_tokens(word, clean_review) %>%
  anti_join(stop_words) %>%
  group_by(sentiment_label) %>%
  summarise(
    total_words = n(),
    unique_words = n_distinct(word),
    avg_word_length = mean(nchar(word))
  )

cat("\nSentiment-wise Word Statistics:\n")
print(sentiment_summary)

# Continue with the original sentiment analysis code from previous script...
# [The rest of the original sentiment analysis code goes here]