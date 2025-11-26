library(data.table)
library(caTools)
library(ggplot2)
library(patchwork)
library(scales)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

dt <- fread("misinformation_samples.csv")

# Data Preprocessing ===========================================================

# Check for missing or duplicated values
summary(dt)
colSums(is.na(dt))
sum(duplicated(dt))

# Check data types and recode to the correct class
sapply(dt, class)

dt$platform <- factor(dt$platform)
dt$month <- factor(dt$month)
dt$weekday <- factor(dt$weekday)
dt$country <- factor(dt$country)
dt$author_verified <- factor(dt$author_verified, levels = c(0, 1), labels = c("No", "Yes"))
dt$is_misinformation <- factor(dt$is_misinformation, levels = c(0, 1), labels = c("No", "Yes"))

# Drop columns not being used in the models
drop_cols <- c("id", "author_id", "timestamp", "date", "time", "city", "timezone")

# Exploratory Data Analysis ====================================================

# Number of misinformation vs not-misinformation posts
ggplot(dt, aes(x = factor(is_misinformation), fill = factor(is_misinformation))) +
  geom_bar() +
  labs(
    title = "Misinformation vs Not-Misinformation Posts",
    x = "Misinformation",
    y = "Count of Posts",
    fill = "Category"
  )

# Misinformation vs platform stacked bar chart (proportion)
p1 <- ggplot(dt, aes(x = platform, fill = factor(is_misinformation))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Proportion of Misinformation Posts by Platform",
    x = "Platform",
    y = "Proportion",
    fill = "Misinformation"
  )

p1

# Misinformation vs month stacked bar chart (proportion)
p2 <- ggplot(dt, aes(x = month, fill = factor(is_misinformation))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_x_discrete(limits = c("January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December")) +
  labs(
    title = "Proportion of Misinformation Posts by Month",
    x = "Month",
    y = "Proportion",
    fill = "Misinformation"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2

# Misinformation vs weekday stacked bar chart (proportion)
p3 <- ggplot(dt, aes(x = weekday, fill = factor(is_misinformation))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_x_discrete(limits = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")) +
  labs(
    title = "Proportion of Misinformation Posts by Weekday",
    x = "Weekday",
    y = "Proportion",
    fill = "Misinformation"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p3

# Misinformation vs country stacked bar chart (proportion)
p4 <- ggplot(dt, aes(x = country, fill = factor(is_misinformation))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Proportion of Misinformation Posts by Country",
    x = "Platform",
    y = "Proportion",
    fill = "Misinformation"
  )

p4

# Combined stacked bar charts
combined_bar <- (p1 | p2) / (p3 | p4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

combined_bar

# Misinformation vs sentiment score boxplot
b1 <- ggplot(dt, aes(x = factor(is_misinformation), y = sentiment_score, fill = factor(is_misinformation))) +
  geom_boxplot() +
  labs(
    title = "Distribution of Sentiment Score by Post Type",
    x = "Misinformation",
    y = "Sentiment Score"
  ) +
  theme(legend.position = "none")

b1

# Misinformation vs toxicity score boxplot
b2 <- ggplot(dt, aes(x = factor(is_misinformation), y = toxicity_score, fill = factor(is_misinformation))) +
  geom_boxplot() +
  labs(
    title = "Distribution of Toxicity Score by Post Type",
    x = "Misinformation",
    y = "Toxicity Score"
  ) +
  theme(legend.position = "none")

b2

# Misinformation vs engagement boxplot
b3 <- ggplot(dt, aes(x = factor(is_misinformation), y = engagement, fill = factor(is_misinformation))) +
  geom_boxplot() +
  labs(
    title = "Distribution of Engagement by Post Type",
    x = "Misinformation",
    y = "Engagement"
  ) +
  theme(legend.position = "none")

b3

# Combined boxplots 1
combined_box1 <- (b1 | b2 | b3)

combined_box1

# Misinformation vs external factchecks count boxplot
b4 <- ggplot(dt, aes(x = factor(is_misinformation), y = external_factchecks_count, fill = factor(is_misinformation))) +
  geom_boxplot() +
  labs(
    title = "Distribution of External Factchecks Count by Post Type",
    x = "Misinformation",
    y = "External Factchecks Count"
  ) +
  theme(legend.position = "none")

b4

# Misinformation vs source domain reliability boxplot
b5 <- ggplot(dt, aes(x = factor(is_misinformation), y = source_domain_reliability, fill = factor(is_misinformation))) +
  geom_boxplot() +
  labs(
    title = "Distribution of Source Domain Reliability by Post Type",
    x = "Misinformation",
    y = "Source Domain Reliability"
  ) +
  theme(legend.position = "none")

b5

# Combined boxplots 2
combined_box2 <- (b4 | b5)

combined_box2

# Train-test split =============================================================
set.seed(2)

# Create 70-30 train-test split and remove unused columns
train <- sample.split(dt$is_misinformation, SplitRatio = 0.7)
trainset <- dt[train == TRUE, ]
testset <- dt[train == FALSE, ]

trainset <- trainset %>% select(-all_of(drop_cols))
testset  <- testset %>% select(-all_of(drop_cols))

# Logistic regression ==========================================================

logistic <- glm(is_misinformation ~ ., data = trainset, family = binomial)
summary(logistic)

log_pred <- predict(logistic, newdata = testset, type = "response")
log_pred_lab <- ifelse(log_pred > 0.5, "Yes", "No")
log_pred_fac <- factor(log_pred_lab, levels = levels(testset$is_misinformation))

confusionMatrix(log_pred_fac, testset$is_misinformation)

# Odds ratio of significant variables
exp(coef(logistic)[summary(logistic)$coefficients[,4] < 0.05])

# CART =========================================================================
cart <- rpart(is_misinformation ~ ., data = trainset, method = 'class', cp = 0)
printcp(cart, digits = 3)
plotcp(cart)

rpart.plot(cart, 
           type = 2,       
           extra = 104,
           fallen.leaves = TRUE,
           )

optimal_cp <- cart$cptable[which.min(cart$cptable[,"xerror"]),"CP"]
pruned_cart <- prune(cart, cp = optimal_cp)

rpart.plot(pruned_cart, 
           type = 2,       
           extra = 104,
           fallen.leaves = TRUE,
)

cart_pred <- predict(pruned_cart, newdata = testset, type = "class")

confusionMatrix(cart_pred, testset$is_misinformation)

# Variable importance
pruned_cart$variable.importance

# Random forest ================================================================
rf_train <- randomForest(is_misinformation ~ ., 
                                  data = trainset, 
                                  importance = TRUE)
rf_train

rf_var_impt <- importance(rf_train)
rf_var_impt

rf_predict <- predict(rf_train, newdata = testset)
rf_predict

confusionMatrix(rf_predict, testset$is_misinformation)

# OOB error and mean decrease accuracy
plot(rf_train)
importance(rf_train)