# 🔍 Misinformation Detection using Machine Learning Models

This project showcases three classical machine learning models trained in R to predict the occurrence of online misinformation.

---

## Exploratory Data Analysis

An initial exploration of the dataset provides a clearer understanding of its structure and key trends.

<br>

**Proportion of Misinformation Across Platforms, Months, Weekdays and Countries**

![Misinformation Proportion](misinformation_proportion.png)

Overall, misinformation and non-misinformation posts are roughly evenly distributed across platforms, months, weekdays, and countries. Twitter, May, Friday, and Brazil show slightly higher proportions. The roughly 50% share across these factors suggests that misinformation is widespread and that these variables alone may not strongly predict it.

<br>

**Distribution of Sentiment, Toxicity, and Engagement Across Posts**

![Sentiment, Toxicity, Engagement Distribution](sentiment_toxicity_engagement_distribution.png)

Examining sentiment, toxicity, and engagement highlights differences between misinformation and non-misinformation posts. Non-misinformation posts generally have more positive language (median sentiment > 0), while misinformation posts tend to be more negative. Surprisingly, non-misinformation posts show slightly higher toxicity. Engagement levels are similar across both, with non-misinformation posts having a slightly higher median.

---

## Machine Learning Models

After the exploratory data analysis, the dataset was cleaned by recoding data types and removing unused columns. A 70/30 train test split was done and the three models were trained and evaluated on the test set.

<br>

| Model                              | Model Complexity          | Accuracy      |
|------------------------------------|---------------------------|---------------|
| Logistic Regression                | 38 X Variables            | 88.0%         |
| Classification and Regression Tree | 4 Terminal Nodes          | 92.7%         |
| Random Forest                      | 500 Trees & RSF Size 4    | 93.3%         |

<br>

Overall, the Random Forest model achieved the highest accuracy for this dataset and predictor variables, with the CART model performing similarly. Logistic Regression had the lowest accuracy and comparatively higher error, though it still provided reasonable predictive performance.
