library(tidyverse)
library(glmnet)
library(e1071)
library(imputeTS)
library(Metrics)

setwd("/Users/alexrabeau/Desktop/SPHERE/NHS-AD-forecasting")

# ============================================================================
# 1. LOAD DATA + PREPROCESSING
# ============================================================================

data <- read.csv("data/turingAI_forecasting_challenge_dataset.csv") # outcome = 'estimated_avoidable_deaths NHS Bristol'
data$date <- as.Date(data$dt, format = "%Y-%m-%d")

forecasting_df <- data %>% 
  dplyr::select(-coverage, -coverage_label, -variable_type, -dt) %>% 
  group_by(date, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>% ungroup() %>% #aggregating to daily resolution
  pivot_wider(id_cols = date, names_from = metric_name, values_from = value,
    names_sep = "_")


# Clean and abbreviate column names
cols_to_abbrev <- names(forecasting_df)[!names(forecasting_df) %in% c("date", "estimated_avoidable_deaths")]
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
cols_to_use <- names(forecasting_df)[!names(forecasting_df) %in% c("date", "estimated_avoidable_deaths")]
abbrev_df <- data.frame(
  original_name = cols_to_abbrev,
  new_name = cols_to_use,
  stringsAsFactors = FALSE
)


# Remove missing values using Kalman imputation
forecasting_df <- forecasting_df %>%
  mutate(across(
    where(is.numeric) & !all_of(c("estimated_avoidable_deaths", "date")),
    ~ na_kalman(.)
  ))

forecasting_df <- na.omit(forecasting_df)


# ============================================================================
# 2. ADDRESS TEMPORAL AUTOCORRELATION
# ============================================================================

predictors = setdiff(colnames(forecasting_df), c("date","estimated_avoidable_deaths"))

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

# Add calendar features (day of week, month, etc.)
forecasting_df <- forecasting_df %>%
  mutate(
    day_of_week = as.numeric(format(date, "%u")),  # 1=Monday, 7=Sunday
    day_of_month = as.numeric(format(date, "%d")),
    month = as.numeric(format(date, "%m")),
    is_weekend = as.numeric(day_of_week %in% c(6, 7)),
    is_monday = as.numeric(day_of_week == 1)
  )

# Remove rows with NA from lagging/rolling
forecasting_df <- na.omit(forecasting_df)

# Update predictors list
predictors <- setdiff(names(forecasting_df), c("date","estimated_avoidable_deaths"))

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

# Forecast horizons
horizons <- 1:10
horizon_results <- list()

set.seed(42)
for (h in horizons) {
  cat("\n--- Horizon:", h, "days ---\n")
  
  # Create horizon-specific dataset
  forecast_h <- forecasting_df %>%
    arrange(date) %>%
    mutate(estimated_avoidable_deaths_future = lead(estimated_avoidable_deaths, n = h)) %>%
    filter(!is.na(estimated_avoidable_deaths_future))
  
  # Time-series split 70/30
  split_h <- floor(0.7 * nrow(forecast_h))
  train_h <- forecast_h[1:split_h, ]
  test_h <- forecast_h[(split_h + 1):nrow(forecast_h), ]
  
  # Scale predictors using training parameters
  scaling_params_h <- list()
  for (col in predictors) {
    if (is.numeric(train_h[[col]])) {
      scaling_params_h[[col]] <- list(
        center = mean(train_h[[col]], na.rm = TRUE),
        scale = sd(train_h[[col]], na.rm = TRUE)
      )
      train_h[[col]] <- scale(train_h[[col]],
                              center = scaling_params_h[[col]]$center,
                              scale = scaling_params_h[[col]]$scale)
      test_h[[col]] <- scale(test_h[[col]],
                             center = scaling_params_h[[col]]$center,
                             scale = scaling_params_h[[col]]$scale)
    }
  }

  # Prepare matrices
  X_train_h <- as.matrix(train_h[, predictors])
  y_train_h <- as.numeric(as.character(train_h$estimated_avoidable_deaths_future))
  X_test_h <- as.matrix(test_h[, predictors])
  y_test_h <- as.numeric(as.character(test_h$estimated_avoidable_deaths_future))
  
  # Cross-validation to find optimal lambda
  cv_enet_h <- cv.glmnet(
    x = X_train_h,
    y = y_train_h,
    family = "gaussian",
    alpha = 0.5,
    nfolds = 10,
    type.measure = "mse"
  )
  
  # Fit horizon-specific Elastic Net model
  enet_model_h <- glmnet(
    x = X_train_h,
    y = y_train_h,
    family = "gaussian",
    alpha = 0.5,
    lambda = cv_enet_h$lambda.min
  )
  
  # Predict values on test set
  test_pred_h <- predict(enet_model_h, X_test_h, type = "response", s = cv_enet_h$lambda.min)
  
  # Evaluate MSE
  mse_h  <- mean((y_test_h - test_pred_h)^2)
  cat("Test MSE:", round(mse_h, 4), "\n")
  
  # Get non-zero coefficients
  coef_enet_h <- coef(enet_model_h, s = cv_enet_h$lambda.min)
  non_zero_coef_h <- coef_enet_h[coef_enet_h[,1] != 0, , drop = FALSE]
  cat("Non-zero coefficients:", nrow(non_zero_coef_h) - 1, "\n")  # -1 for intercept

  # True future forecast using last row
  X_last <- as.matrix(test_h[nrow(test_h), predictors, drop = FALSE])
  future_forecast <- predict(enet_model_h, X_last, type = "response", s = cv_enet_h$lambda.min)
  cat("Forecast for last row (", h, "days ahead):", round(future_forecast, 2), "\n")
  
  # Store results
  horizon_results[[as.character(h)]] <- list(
    model = enet_model_h,
    mse = mse_h,
    predictions = test_pred_h,
    actual = y_test_h,
    lambda = cv_enet_h$lambda.min,
    non_zero_coef = non_zero_coef_h,
    future_forecast_last_row = future_forecast
  )

  }


# ============================================================================
# 5. OUTPUT
# ============================================================================

# Extract MSE values from all horizons into a named numeric vector
mse_values <- sapply(horizon_results, function(x) x$mse)

# Compute mean MSE for horizons 1–5 and 6–10
mean_mse_1_5  <- mean(mse_values[as.character(1:5)],  na.rm = TRUE)
mean_mse_6_10 <- mean(mse_values[as.character(6:10)], na.rm = TRUE)

cat("Mean MSE (Days 1–5):", round(mean_mse_1_5, 4), "\n")
cat("Mean MSE (Days 6–10):", round(mean_mse_6_10, 4), "\n")


#Summary table
forecast_summary <- data.frame(horizon = integer(), test_mse = numeric(), future_forecast_last_row = numeric())
for (h in horizons) {
  res <- horizon_results[[as.character(h)]]
  forecast_summary <- rbind(
    forecast_summary,
    data.frame(
      horizon = h,
      test_mse = round(res$mse, 4),
      future_forecast_last_row = round(as.numeric(res$future_forecast_last_row), 2)
    )
  )
}

# Write to CSV
# write.csv(forecast_summary, "horizon_forecast_summary.csv", row.names = FALSE)
