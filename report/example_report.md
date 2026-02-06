---
output:
  html_document: default
  pdf_document: default
---
# NHS System Pressure Forecasting

**Author:** Alex Rabeau  
**Date:** February 2026

## Introduction

This report outlines a workflow for forecasting estimated avoidable deaths in NHS trust-level data using an Elastic Net regression model. The pipeline includes data loading, preprocessing, feature engineering, handling skewness, multi-horizon forecasting, and future prediction generation.

The dataset consists of daily NHS system metrics, including hospital, ambulance, and performance indicators. The target variable is estimated_avoidable_deaths.


## 1. Load and Preprocess Data
We begin by loading the combined outcome and predictor datasets. Missing values were imputed using Kalman smoothing, column names were cleaned and abbreviated, and data was aggregated to a daily resolution.

```r
library(tidyverse)
library(imputeTS)
library(glmnet)
library(e1071)
library(Metrics)

# Load dataset
data <- read.csv("data/turingAI_forecasting_challenge_dataset.csv")
data$date <- as.Date(data$dt, format = "%Y-%m-%d")

# Aggregate daily and pivot
forecasting_df <- data %>%
  select(-coverage, -coverage_label, -variable_type, -dt) %>%
  group_by(date, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(id_cols = date, names_from = metric_name, values_from = value)

# Abbreviate and clean column names
cols_to_abbrev <- setdiff(names(forecasting_df), c("date", "estimated_avoidable_deaths"))
abbrev_names <- make.names(abbreviate(cols_to_abbrev, minlength = 8), unique = TRUE)
names(forecasting_df)[names(forecasting_df) %in% cols_to_abbrev] <- abbrev_names

clean_names <- colnames(forecasting_df) %>%
  gsub("[0-9]", "", .) %>%
  gsub("[()]", "", .) %>%
  gsub("[ -]", "_", .) %>%
  gsub("%", "pct", .) %>%
  gsub("[^[:alnum:]_]", "", .) %>%
  tolower()
colnames(forecasting_df) <- make.names(clean_names, unique = TRUE)

# Kalman imputation for numeric predictors
predictors <- setdiff(names(forecasting_df), c("date","estimated_avoidable_deaths"))
forecasting_df <- forecasting_df %>%
  mutate(across(all_of(predictors), ~ na_kalman(.))) %>%
  na.omit()
```

## 2. Feature Engineering: Rolling and Calendar Features

To handle temporal autocorrelation, we create rolling (7-day) features for all numeric predictors, and add calendar features to capture weekday/month effects.

```r
create_rolling_features <- function(data, vars, windows = c(7)) {
  result <- data
  for(var in vars) {
    for(window in windows) {
      result[[paste0(var, "_roll_mean_", window)]] <-
        zoo::rollmean(data[[var]], k = window, fill = NA, align = "right")
      result[[paste0(var, "_roll_sd_", window)]] <-
        zoo::rollapply(data[[var]], width = window, FUN = sd, fill = NA, align = "right")
    }
  }
  return(result)
}

forecasting_df <- create_rolling_features(forecasting_df, predictors, windows = c(7))

forecasting_df <- forecasting_df %>%
  mutate(
    day_of_week = as.numeric(format(date, "%u")),
    day_of_month = as.numeric(format(date, "%d")),
    month = as.numeric(format(date, "%m")),
    is_weekend = as.numeric(day_of_week %in% c(6,7)),
    is_monday = as.numeric(day_of_week == 1)
  ) %>%
  na.omit()
```

## 3. Handling Skewness

Numeric predictors with skew > 1 were transformed:

- **Positive skew:** log1p (if all values > 0) or sqrt  
- **Negative skew:** squared

```r
skewness_results <- data.frame(variable=character(), original_skewness=numeric(), transformation=character(), stringsAsFactors=FALSE)

for (col in predictors) {
  x <- forecasting_df[[col]]
  if (is.numeric(x)) {
    skew_val <- e1071::skewness(x, na.rm = TRUE)
    transformation <- "none"
    if (abs(skew_val) > 1) {
      if (skew_val > 1 && all(x > 0, na.rm = TRUE)) {
        forecasting_df[[col]] <- log1p(x)
        transformation <- "log1p"
      } else if (skew_val > 1) {
        forecasting_df[[col]] <- sqrt(x - min(x, na.rm=TRUE) + 1)
        transformation <- "sqrt"
      } else if (skew_val < -1) {
        forecasting_df[[col]] <- x^2
        transformation <- "squared"
      }
    }
    skewness_results <- rbind(skewness_results, data.frame(variable=col, original_skewness=skew_val, transformation=transformation))
  }
}
```

## 4. Multi-Horizon Forecasting
Elastic Net regression models were trained for 1–10 day horizons using time-based 70/30 train-test splits. Model hyperparameters were tuned via 10-fold CV minimizing MSE. Models were evaluated using MSE.

```r
horizons <- 1:10
horizon_results <- list()

set.seed(42)
for (h in horizons) {
  forecast_h <- forecasting_df %>%
    arrange(date) %>%
    mutate(estimated_avoidable_deaths_future = lead(estimated_avoidable_deaths, n = h)) %>%
    filter(!is.na(estimated_avoidable_deaths_future))
  
  split_h <- floor(0.7 * nrow(forecast_h))
  train_h <- forecast_h[1:split_h, ]
  test_h <- forecast_h[(split_h+1):nrow(forecast_h), ]
  
  # Scale predictors
  for (col in predictors) {
    train_h[[col]] <- scale(train_h[[col]], center=mean(train_h[[col]]), scale=sd(train_h[[col]]))
    test_h[[col]] <- scale(test_h[[col]], center=mean(train_h[[col]]), scale=sd(train_h[[col]]))
  }
  
  # Fit Elastic Net
  X_train_h <- as.matrix(train_h[, predictors])
  y_train_h <- train_h$estimated_avoidable_deaths_future
  X_test_h <- as.matrix(test_h[, predictors])
  y_test_h <- test_h$estimated_avoidable_deaths_future
  
  cv_enet_h <- cv.glmnet(X_train_h, y_train_h, alpha=0.5, nfolds=10, type.measure="mse")
  enet_model_h <- glmnet(X_train_h, y_train_h, alpha=0.5, lambda=cv_enet_h$lambda.min)
  
  test_pred_h <- predict(enet_model_h, X_test_h, s=cv_enet_h$lambda.min)
  mse_h <- mean((y_test_h - test_pred_h)^2)
  
  horizon_results[[as.character(h)]] <- list(
    model = enet_model_h,
    mse = mse_h,
    predictions = test_pred_h,
    actual = y_test_h,
    future_forecast_last_row = predict(enet_model_h, as.matrix(test_h[nrow(test_h), predictors, drop=FALSE]), s=cv_enet_h$lambda.min)
  )
}
```

Forecast performance:

| Horizon |   MSE  | AD Forecast |
|:-------:|:------:|:-----------:|
|    1    | 0.0619 |    0.93     |
|    2    | 0.1265 |    1.10     |
|    3    | 0.2193 |    1.12     |
|    4    | 0.2011 |    1.03     |
|    5    | 0.1481 |    1.02     |
|    6    | 0.1832 |    1.11     |
|    7    | 0.3347 |    1.17     |
|    8    | 0.1937 |    0.95     |
|    9    | 0.2037 |    1.05     |
|   10    | 0.1779 |    1.19     |

**Mean MSE (Days 1–5):** 0.1572  
**Mean MSE (Days 6–10):** 0.227

