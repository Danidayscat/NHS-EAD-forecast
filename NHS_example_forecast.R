library(tidyverse)
library(glmnet)
library(e1071)
library(imputeTS)
library(zoo)

setwd("/Users/alexrabeau/Desktop/SPHERE/NHS-AD-forecasting")

# ============================================================================
# 1. LOAD DATA + PREPROCESSING
# ============================================================================

data <- read.csv("data/turingAI_forecasting_challenge_dataset.csv") # outcome = 'estimated_avoidable_deaths NHS Bristol'

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

# data cleaning
forecasting_df <- data %>%
  mutate(midday_day = if_else(
      format(dt, "%H:%M:%S") <= "12:00:00", date,date + 1)) %>% #setting midday threshold for downstream forecasts
  dplyr::select(-coverage, -coverage_label, -variable_type, -dt, -date, -time) %>% 
  group_by(midday_day, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>% #aggregating to daily temporal resolution to match outcome variable
  pivot_wider(id_cols = midday_day, names_from = metric_name, values_from = value,
    names_sep = "_")

# clean and abbreviate column names
cols_to_abbrev <- names(forecasting_df)[!names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths")]
abbrev_names <- make.names(abbreviate(cols_to_abbrev, minlength = 8), unique = TRUE)
names(forecasting_df)[names(forecasting_df) %in% cols_to_abbrev] <- abbrev_names

clean_names <- colnames(forecasting_df) %>%
  gsub("[0-9]", "", .) %>%
  gsub("[()]", "", .) %>%
  gsub("[ -]", "_", .) %>%
  gsub("%", "pct", .) %>%
  gsub("[^[:alnum:]_]", "", .)

clean_names <- tolower(clean_names)
colnames(forecasting_df) <- make.names(clean_names, unique = TRUE)

# Store mapping for reference
cols_to_use <- names(forecasting_df)[!names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths")]
abbrev_df <- data.frame(
  original_name = cols_to_abbrev,
  new_name = cols_to_use,
  stringsAsFactors = FALSE
)

# Remove missing values using Kalman imputation
forecasting_df <- forecasting_df %>%
  mutate(across(
    where(is.numeric) & !all_of(c("estimated_avoidable_deaths", "midday_day")),
    ~ na_kalman(.)
  ))

forecasting_df <- na.omit(forecasting_df)


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

predictors = setdiff(colnames(forecasting_df), c("midday_day","estimated_avoidable_deaths"))

# Function to create rolling window features
create_rolling_features <- function(data, vars, windows = c(7)) {
  result <- data
  for(var in vars) {
    for(window in windows) {
      # Rolling mean
      result[[paste0(var, "_roll_mean_", window)]] <- 
        zoo::rollmean(data[[var]], k = window, fill = NA, align = "right")
      
      # Rolling std dev
      result[[paste0(var, "_roll_sd_", window)]] <- 
        zoo::rollapply(data[[var]], width = window, FUN = sd, fill = NA, align = "right")
    }
  }
  return(result)
}

# Create rolling features (7-day window for weekly patterns)
forecasting_df <- create_rolling_features(forecasting_df, predictors, windows = c(7))


# Lagged Y as predictor
forecasting_df <- forecasting_df %>%
  mutate(estimated_avoidable_deaths_lag3 = lag(estimated_avoidable_deaths, 3)) %>%
  na.omit()


# Update predictors list
predictors <- setdiff(names(forecasting_df), c("midday_day","estimated_avoidable_deaths"))


# ============================================================================
# 3. HANDLE SKEWNESS WITH TRANSFORMATIONS
# ============================================================================

skewness_results <- data.frame(
  variable = character(),
  original_skewness = numeric(),
  transformation = character(),
  stringsAsFactors = FALSE
)

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
        forecasting_df[[col]] <- sqrt(x - min(x, na.rm = TRUE) + 1)
        transformation <- "sqrt"
      } else if (skew_val < -1) {
        forecasting_df[[col]] <- x^2
        transformation <- "squared"
      }
    }
    
    skewness_results <- rbind(skewness_results, data.frame(
      variable = col,
      original_skewness = skew_val,
      transformation = transformation
    ))
  }
}


# ============================================================================
# 4. MULTI-HORIZON FORECASTING WITH ELASTIC NET
# ============================================================================

# Parameters
set.seed(123)
n <- nrow(forecasting_df)
train_window <- 90
horizon <- 10
n_forecasts <- n - (train_window + horizon) + 1  
alpha_val <- 0.5  

# Storage
pred_matrix  <- matrix(NA, nrow = n_forecasts, ncol = horizon)
actual_matrix <- matrix(NA, nrow = n_forecasts, ncol = horizon)

# Rolling window forecasting
for (i in 1:n_forecasts) {
  
  cat("\n--- Forecast:", i)
  
  # Index ranges
  train_idx <- i:(i + train_window - 1)
  test_idx  <- (i + train_window):(i + train_window + horizon - 1)
  
  # Split data
  train_data <- forecasting_df[train_idx, ]
  test_data  <- forecasting_df[test_idx, ]
  
  # Keep only predictors with non-zero SD in training
  train_data <- train_data[, sapply(train_data, sd) != 0, drop = FALSE]
  predictors <- setdiff(names(train_data), c("midday_day","estimated_avoidable_deaths"))

  # Scale predictors using training parameters
  scaling_params <- list()
  for (col in predictors) {
    if (is.numeric(train_data[[col]])) {
      scaling_params[[col]] <- list(
        center = mean(train_data[[col]], na.rm = TRUE),
        scale = sd(train_data[[col]], na.rm = TRUE)
      )
      train_data[[col]] <- scale(train_data[[col]],
                              center = scaling_params[[col]]$center,
                              scale = scaling_params[[col]]$scale)
      test_data[[col]] <- scale(test_data[[col]],
                             center = scaling_params[[col]]$center,
                             scale = scaling_params[[col]]$scale)
    }
  }
  
  # Prepare matrices
  X_train <- as.matrix(train_data[, predictors])
  y_train <- train_data$estimated_avoidable_deaths
  X_test <- as.matrix(test_data[, predictors])

  # Cross-validation to find optimal lambda
  cv_enet <- cv.glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    alpha = alpha_val,
    nfolds = 10,
    type.measure = "mse"
  )
  
  # Fit Elastic Net model
  enet_model <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    alpha = alpha_val,
    lambda = cv_enet$lambda.min
  )
  
  # Predict values on test set
  test_pred <- predict(enet_model, X_test, type = "response", s = cv_enet$lambda.min)
  
  # Store predictions and actuals
  pred_matrix[i, ] <- test_pred
  actual_matrix[i, ] <- test_data$estimated_avoidable_deaths
  
}


# ============================================================================
# 5. OUTPUT
# ============================================================================

# 1. Predicted matrix
pred_out <- as.data.frame(pred_matrix)
colnames(pred_out) <- paste0("day_", 1:horizon)
pred_out$forecast_id <- 1:n_forecasts
pred_out <- pred_out[, c("forecast_id", paste0("day_", 1:horizon))]

# Write to CSV
write.csv(pred_out, "pred_matrix.csv", row.names = FALSE)


# 2. MSE summary
# Define helper
mse <- function(a, p) mean((a - p)^2, na.rm = TRUE)
mse_df <- data.frame(forecast_id = 1:n_forecasts, mse_1_5 = "", mse_6_10 = "")

for (i in 1:n_forecasts) {
  # Actual vs predicted for this forecast
  a <- actual_matrix[i, ]
  p <- pred_matrix[i, ]
  
  # Compute MSE for horizons 1–5 and 6–10
  mse_df$mse_1_5[i] <- mse(a[1:5],  p[1:5])
  mse_df$mse_6_10[i] <- mse(a[6:10], p[6:10])
}

write.csv(mse_df, "mse_summary.csv", row.names = FALSE)

