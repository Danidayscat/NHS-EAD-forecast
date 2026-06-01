# ============================================================
# run_forecast_xgb.R
# Standalone script to prepare data, engineer features,
# run rolling-origin XGBoost validation, and export outputs
# for the final report and competition submission.
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(e1071)
  library(imputeTS)
  library(zoo)
  library(here)
  library(xgboost)
})

set.seed(666)

# ------------------------------------------------------------
# 0. Paths and data input
# ------------------------------------------------------------
data_dir <- here("data")
csv_path <- here("data", "turingAI_forecasting_challenge_dataset.csv")
zip_path <- here("data", "turingAI_forecasting_challenge_dataset.csv.zip")

if (!dir.exists(data_dir)) {
  stop("Directory /data was not found.")
}

if (!file.exists(csv_path)) {
  if (file.exists(zip_path)) {
    unzip(zip_path, exdir = data_dir)
  } else {
    stop("Neither CSV nor ZIP development dataset was found in /data.")
  }
}

# ------------------------------------------------------------
# 1. Load and preprocess development data
# ------------------------------------------------------------
raw_data <- read.csv(csv_path)

raw_data <- raw_data %>%
  mutate(
    dt = parse_date_time(dt, orders = c("Ymd HMS", "Ymd"), tz = "UTC"),
    date = as.Date(dt),
    time = format(dt, "%H:%M:%S")
  ) %>%
  filter(dt <= as.POSIXct("2025-09-30 00:00:00", tz = "UTC"))

forecasting_df <- raw_data %>%
  mutate(
    midday_day = if_else(
      format(dt, "%H:%M:%S") <= "12:00:00",
      date,
      date + 1
    )
  ) %>%
  select(-coverage, -coverage_label, -variable_type, -dt, -date, -time) %>%
  group_by(midday_day, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    id_cols = midday_day,
    names_from = metric_name,
    values_from = value,
    names_sep = "_"
  ) %>%
  arrange(midday_day)

# Clean names
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

num_cols <- forecasting_df %>%
  select(where(is.numeric)) %>%
  names() %>%
  setdiff("estimated_avoidable_deaths")

# ------------------------------------------------------------
# 2. Missing-value handling
# ------------------------------------------------------------
impute_safe_ts <- function(x) {
  n_total <- length(x)
  n_na <- sum(is.na(x))
  n_valid <- sum(is.finite(x))
  
  if (n_valid < 3 || n_na / n_total > 0.5) {
    if (n_valid > 0) {
      med <- median(x, na.rm = TRUE)
      if (is.na(med)) med <- mean(x, na.rm = TRUE)
      x[is.na(x)] <- med
    }
    return(x)
  }
  
  sdx <- sd(x, na.rm = TRUE)
  if (!is.finite(sdx) || sdx == 0) return(x)
  
  out <- tryCatch(
    imputeTS::na_kalman(x),
    error = function(e) {
      tryCatch(
        imputeTS::na_interpolation(x, option = "linear"),
        error = function(e2) {
          med <- median(x, na.rm = TRUE)
          if (is.na(med)) med <- mean(x, na.rm = TRUE)
          if (is.na(med)) return(x)
          x[is.na(x)] <- med
          x
        }
      )
    }
  )
  
  out
}

forecasting_df <- forecasting_df %>%
  mutate(across(all_of(num_cols), impute_safe_ts))

# ------------------------------------------------------------
# 3. Feature engineering
# ------------------------------------------------------------
forecasting_df <- forecasting_df %>%
  mutate(
    dow = wday(midday_day, week_start = 1),
    dom = mday(midday_day),
    month = month(midday_day),
    week = isoweek(midday_day),
    is_weekend = if_else(dow %in% c(6, 7), 1, 0),
    sin_dow = sin(2 * pi * dow / 7),
    cos_dow = cos(2 * pi * dow / 7),
    sin_month = sin(2 * pi * month / 12),
    cos_month = cos(2 * pi * month / 12)
  )

create_rolling_features <- function(data, vars, windows = c(3, 7)) {
  result <- data
  for (var in vars) {
    for (window in windows) {
      result[[paste0(var, "_roll_mean_", window)]] <-
        zoo::rollapply(data[[var]], width = window, FUN = mean, fill = NA, align = "right", na.rm = TRUE)
      result[[paste0(var, "_roll_sd_", window)]] <-
        zoo::rollapply(data[[var]], width = window, FUN = sd, fill = NA, align = "right")
    }
  }
  result
}

base_predictors <- setdiff(names(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))
forecasting_df <- create_rolling_features(forecasting_df, base_predictors, windows = c(3, 7))

forecasting_df <- forecasting_df %>%
  mutate(
    y_lag_3  = lag(estimated_avoidable_deaths, 3),
    y_lag_4  = lag(estimated_avoidable_deaths, 4),
    y_lag_5  = lag(estimated_avoidable_deaths, 5),
    y_lag_7  = lag(estimated_avoidable_deaths, 7),
    y_lag_10 = lag(estimated_avoidable_deaths, 10),
    y_lag_14 = lag(estimated_avoidable_deaths, 14),
    y_roll_mean_7  = lag(zoo::rollapply(estimated_avoidable_deaths, 7, mean, fill = NA, align = "right", na.rm = TRUE), 3),
    y_roll_sd_7    = lag(zoo::rollapply(estimated_avoidable_deaths, 7, sd, fill = NA, align = "right"), 3),
    y_roll_mean_14 = lag(zoo::rollapply(estimated_avoidable_deaths, 14, mean, fill = NA, align = "right", na.rm = TRUE), 3),
    y_roll_sd_14   = lag(zoo::rollapply(estimated_avoidable_deaths, 14, sd, fill = NA, align = "right"), 3)
  )

final_predictors <- setdiff(names(forecasting_df), c("midday_day", "estimated_avoidable_deaths"))

# ------------------------------------------------------------
# 4. Skewness-based transformations
# ------------------------------------------------------------
skewness_results <- data.frame(
  variable = character(),
  original_skewness = numeric(),
  transformation = character(),
  stringsAsFactors = FALSE
)

for (col in final_predictors) {
  x <- forecasting_df[[col]]
  if (is.numeric(x)) {
    skew_val <- e1071::skewness(x, na.rm = TRUE)
    transformation <- "none"
    
    if (is.finite(skew_val) && abs(skew_val) > 1) {
      if (skew_val > 1 && all(x[is.finite(x)] > 0)) {
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
    
    skewness_results <- rbind(
      skewness_results,
      data.frame(
        variable = col,
        original_skewness = skew_val,
        transformation = transformation
      )
    )
  }
}

forecasting_df <- forecasting_df %>% drop_na()

transformation_summary <- skewness_results %>%
  count(transformation, sort = TRUE)

data_summary <- tibble(
  n_days = nrow(forecasting_df),
  n_predictors = ncol(forecasting_df) - 2,
  start_date = min(forecasting_df$midday_day),
  end_date = max(forecasting_df$midday_day)
)

# ------------------------------------------------------------
# 5. Rolling-origin XGBoost
# ------------------------------------------------------------
fit_xgb_one_window <- function(train_data, test_data, target = "estimated_avoidable_deaths") {
  predictors <- setdiff(names(train_data), c("midday_day", target))
  
  keep_predictor <- function(x) {
    if (!is.numeric(x)) return(FALSE)
    sdx <- sd(x, na.rm = TRUE)
    is.finite(sdx) && sdx > 0
  }
  
  predictors <- predictors[sapply(train_data[, predictors, drop = FALSE], keep_predictor)]
  
  X_train <- as.matrix(train_data[, predictors, drop = FALSE])
  y_train <- train_data[[target]]
  X_test  <- as.matrix(test_data[, predictors, drop = FALSE])
  
  X_train[!is.finite(X_train)] <- NA
  X_test[!is.finite(X_test)] <- NA
  
  n_train <- nrow(X_train)
  val_size <- max(20, floor(0.20 * n_train))
  if (val_size >= (n_train - 10)) val_size <- max(10, floor(0.10 * n_train))
  split_idx <- n_train - val_size
  
  dtrain <- xgb.DMatrix(X_train[1:split_idx, , drop = FALSE], label = y_train[1:split_idx], missing = NA)
  dval   <- xgb.DMatrix(X_train[(split_idx + 1):n_train, , drop = FALSE], label = y_train[(split_idx + 1):n_train], missing = NA)
  dtest  <- xgb.DMatrix(X_test, missing = NA)
  
  model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = 0.05,
      max_depth = 5,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    data = dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, eval = dval),
    early_stopping_rounds = 30,
    verbose = 0,
    maximize = FALSE
  )
  
  pred <- as.numeric(predict(model, dtest))
  
  list(
    pred_xgb = pred,
    best_iteration = if (!is.null(model$best_iteration)) as.integer(model$best_iteration) else NA_integer_,
    n_features = as.integer(length(predictors))
  )
}

target <- "estimated_avoidable_deaths"
train_window <- 180
horizon <- 10
n <- nrow(forecasting_df)
n_forecasts <- n - (train_window + horizon) + 1

pred_xgb_mat <- matrix(NA, nrow = n_forecasts, ncol = horizon)
actual_mat <- matrix(NA, nrow = n_forecasts, ncol = horizon)
window_runtime_sec <- numeric(n_forecasts)
best_iterations <- integer(n_forecasts)
n_features_used <- integer(n_forecasts)

for (i in seq_len(n_forecasts)) {
  start_i <- Sys.time()
  
  train_idx <- i:(i + train_window - 1)
  test_idx  <- (i + train_window):(i + train_window + horizon - 1)
  
  train_data <- forecasting_df[train_idx, , drop = FALSE]
  test_data  <- forecasting_df[test_idx, , drop = FALSE]
  
  fit_out <- fit_xgb_one_window(train_data, test_data, target = target)
  
  pred_xgb_mat[i, ] <- fit_out$pred_xgb
  actual_mat[i, ] <- test_data[[target]]
  best_iterations[i] <- fit_out$best_iteration
  n_features_used[i] <- fit_out$n_features
  
  window_runtime_sec[i] <- as.numeric(difftime(Sys.time(), start_i, units = "secs"))
  
  if (i %% 25 == 0 || i == n_forecasts) {
    message(sprintf("Completed %s / %s windows", i, n_forecasts))
  }
}

# ------------------------------------------------------------
# 6. Outputs
# ------------------------------------------------------------
mse_vec <- function(a, p) mean((a - p)^2, na.rm = TRUE)

results_summary <- tibble(
  model = "XGBoost (Final Submission)",
  mse_1_5 = mse_vec(actual_mat[, 1:5], pred_xgb_mat[, 1:5]),
  mse_6_10 = mse_vec(actual_mat[, 6:10], pred_xgb_mat[, 6:10])
)

runtime_summary <- tibble(
  train_window_days = train_window,
  horizon_days = horizon,
  n_forecasts = n_forecasts,
  total_runtime_min = sum(window_runtime_sec) / 60,
  mean_runtime_sec = mean(window_runtime_sec),
  median_runtime_sec = median(window_runtime_sec),
  mean_best_iteration = mean(best_iterations),
  median_best_iteration = median(best_iterations),
  mean_features_used = mean(n_features_used)
)

pred_out <- as.data.frame(pred_xgb_mat)
colnames(pred_out) <- paste0("day_", 1:horizon)
pred_out$forecast_id <- seq_len(n_forecasts)
pred_out <- pred_out[, c("forecast_id", paste0("day_", 1:horizon))]

mse_df <- tibble(
  forecast_id = seq_len(n_forecasts),
  mse_1_5 = rowMeans((actual_mat[, 1:5] - pred_xgb_mat[, 1:5])^2, na.rm = TRUE),
  mse_6_10 = rowMeans((actual_mat[, 6:10] - pred_xgb_mat[, 6:10])^2, na.rm = TRUE)
)

validation_summary <- tibble(
  first_training_day = forecasting_df$midday_day[1],
  first_test_day = forecasting_df$midday_day[train_window + 1],
  last_test_day = forecasting_df$midday_day[n],
  n_forecasts = n_forecasts
)

out_dir <- here("submission")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write.csv(data_summary, file.path(out_dir, "data_summary.csv"), row.names = FALSE)
write.csv(transformation_summary, file.path(out_dir, "transformation_summary.csv"), row.names = FALSE)
write.csv(results_summary, file.path(out_dir, "results_summary.csv"), row.names = FALSE)
write.csv(runtime_summary, file.path(out_dir, "runtime_summary.csv"), row.names = FALSE)
write.csv(pred_out, file.path(out_dir, "pred_matrix.csv"), row.names = FALSE)
write.csv(mse_df, file.path(out_dir, "mse_summary.csv"), row.names = FALSE)
write.csv(validation_summary, file.path(out_dir, "validation_summary.csv"), row.names = FALSE)

message("All output files were created successfully.")