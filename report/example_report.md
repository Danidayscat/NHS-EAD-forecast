---
output:
  pdf_document: default
  html_document: default
---
# NHS Severe Patient Harm Forecasting

**Author:** Alex Rabeau  
**Date:** February 2026

## Introduction

This report outlines a refined workflow for forecasting estimated avoidable deaths in NHS trust-level data using an Elastic Net regression model.  
The updated pipeline includes improved preprocessing (midday aggregation and timestamp handling), feature engineering with lagged outcomes and rolling statistics, skewness correction, and a rolling-window multi-horizon forecast strategy.

The dataset consists of daily NHS system metrics, including hospital, ambulance, and performance indicators.  
The target variable is **estimated_avoidable_deaths**.

---


## 1. Load and Preprocess Data

We begin by loading the combined outcome and predictor datasets.  
Data timestamps are standardized to UTC and aggregated to daily resolution using a midday threshold (entries before 12:00 are assigned to the same day; those after, to the next day).  
Columns are cleaned, abbreviated, and imputed using Kalman smoothing.

```r
library(tidyverse)
library(glmnet)
library(e1071)
library(imputeTS)
library(zoo)

setwd("/Users/alexrabeau/Desktop/SPHERE/NHS-AD-forecasting")

# Load dataset
data <- read.csv("data/turingAI_forecasting_challenge_dataset.csv")

# date formatting
data <- data %>%
  mutate(
    dt = parse_date_time(dt, orders = c("Ymd HMS", "Ymd")),
    dt = force_tz(dt, tzone = "UTC"),
    date = as.Date(dt),
    time = format(dt, "%H:%M:%S")
  )

data <- data %>% 
  filter(dt <= as.POSIXct("2025-09-30 00:00:00", tz = "UTC")) #filter out assessment dataset

# Midday aggregation
forecasting_df <- data %>%
  mutate(midday_day = if_else(format(dt, "%H:%M:%S") <= "12:00:00", date, date + 1)) %>%
  select(-coverage, -coverage_label, -variable_type, -dt, -date, -time) %>%
  group_by(midday_day, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(id_cols = midday_day, names_from = metric_name, values_from = value, names_sep = "_")

# Clean and abbreviate names
cols_to_abbrev <- names(forecasting_df)[!names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths")]
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

# Impute missing numeric predictors
forecasting_df <- forecasting_df %>%
  mutate(across(
    where(is.numeric) & !all_of(c("estimated_avoidable_deaths", "midday_day")),
    ~ na_kalman(.)
  )) %>%
  na.omit()
```


## 2. Feature Engineering: Rolling and Lag Features

We introduce rolling 7-day mean and standard deviation features to capture short-term temporal dependencies,
and a 3-day lag of the target variable to enhance autoregressive structure.

```r
predictors = setdiff(colnames(forecasting_df), c("midday_day","estimated_avoidable_deaths"))

# Rolling features
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

# Add lagged Y
forecasting_df <- forecasting_df %>%
  mutate(estimated_avoidable_deaths_lag3 = lag(estimated_avoidable_deaths, 3)) %>%
  na.omit()

# Update predictor list
predictors <- setdiff(names(forecasting_df), c("midday_day","estimated_avoidable_deaths"))
```


## 3. Handling Skewness

Numeric predictors with skew > 1 were transformed:

- Positive skew: log1p (if all values > 0) or sqrt  
- Negative skew: squared

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


## 4. Rolling Multi-Horizon Forecasting

We train Elastic Net regression models using a rolling 90-day window across the time series, forecasting 10 days ahead at each iteration.
This allows for dynamic model adaptation to evolving NHS system pressures.

Each iteration involves:

- Scaling predictors on the training window
- Cross-validating $\lambda$ (lambda) using 10-fold CV
- Storing predictions and actuals for 10-day horizons

```r
set.seed(123)
n <- nrow(forecasting_df)
train_window <- 90
horizon <- 10
n_forecasts <- n - (train_window + horizon) + 1
alpha_val <- 0.5

pred_matrix  <- matrix(NA, nrow = n_forecasts, ncol = horizon)
actual_matrix <- matrix(NA, nrow = n_forecasts, ncol = horizon)

for (i in 1:n_forecasts) {

  cat("\n--- Forecast:", i)

  train_idx <- i:(i + train_window - 1)
  test_idx  <- (i + train_window):(i + train_window + horizon - 1)

  train_data <- forecasting_df[train_idx, ]
  test_data  <- forecasting_df[test_idx, ]

  train_data <- train_data[, sapply(train_data, sd) != 0, drop = FALSE]
  predictors <- setdiff(names(train_data), c("midday_day","estimated_avoidable_deaths"))

  scaling_params <- list()
  for (col in predictors) {
    if (is.numeric(train_data[[col]])) {
      scaling_params[[col]] <- list(center = mean(train_data[[col]], na.rm = TRUE),
                                    scale = sd(train_data[[col]], na.rm = TRUE))
      train_data[[col]] <- scale(train_data[[col]], center = scaling_params[[col]]$center, scale = scaling_params[[col]]$scale)
      test_data[[col]] <- scale(test_data[[col]], center = scaling_params[[col]]$center, scale = scaling_params[[col]]$scale)
    }
  }

  X_train <- as.matrix(train_data[, predictors])
  y_train <- train_data$estimated_avoidable_deaths
  X_test  <- as.matrix(test_data[, predictors])

  cv_enet <- cv.glmnet(x = X_train, y = y_train, family = "gaussian",
                       alpha = alpha_val, nfolds = 10, type.measure = "mse")

  enet_model <- glmnet(x = X_train, y = y_train, family = "gaussian",
                       alpha = alpha_val, lambda = cv_enet$lambda.min)

  test_pred <- predict(enet_model, X_test, type = "response", s = cv_enet$lambda.min)

  pred_matrix[i, ] <- test_pred
  actual_matrix[i, ] <- test_data$estimated_avoidable_deaths
}
```


## 5. Forecast Output and Evaluation

Forecasts for each rolling iteration and horizon are stored, with MSE computed for short-term (Days 1–5) and medium-term (Days 6–10) horizons.

```r
# Export predicted values
pred_out <- as.data.frame(pred_matrix)
colnames(pred_out) <- paste0("day_", 1:horizon)
pred_out$forecast_id <- 1:n_forecasts
pred_out <- pred_out[, c("forecast_id", paste0("day_", 1:horizon))]
write.csv(pred_out, "pred_matrix.csv", row.names = FALSE)

# Compute MSE summaries
mse <- function(a, p) mean((a - p)^2, na.rm = TRUE)
mse_df <- data.frame(forecast_id = 1:n_forecasts, mse_1_5 = "", mse_6_10 = "")

for (i in 1:n_forecasts) {
  a <- actual_matrix[i, ]
  p <- pred_matrix[i, ]
  mse_df$mse_1_5[i] <- mse(a[1:5],  p[1:5])
  mse_df$mse_6_10[i] <- mse(a[6:10], p[6:10])
}

write.csv(mse_df, "mse_summary.csv", row.names = FALSE)
```