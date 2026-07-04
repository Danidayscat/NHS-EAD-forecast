# ============================================================
# run_forecast_xgb_def.R
# Standalone script to prepare data, engineer features,
# run hyperparameter tuning (rolling-origin), XGBoost validation, 
# and export outputs for the competition submission.
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(imputeTS)
  library(caret)
  library(zoo)
  library(here)
  library(xgboost)
})

set.seed(666)

# ------------------------------------------------------------
# 0. Paths and data input
# ------------------------------------------------------------
data_dir <- here("data")
dev_csv <- here("data", "turingAI_forecasting_challenge_dataset.csv")
val_csv <- here("data", "turingAI_forecasting_challenge_validation_dataset.csv")

if (!dir.exists(data_dir)) stop("Directory /data was not found.")

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) > 0) args[1] else "validation"
mode <- match.arg(mode, c("development", "validation"))

dataset_label <- mode
message("========================================")
message("INICIANDO PIPELINE EN MODO: ", toupper(mode))
message("========================================")

# WARM-START: Combine development and validation datasets
if (mode == "validation") {
  if (!file.exists(val_csv)) stop("CRITICAL ERROR: Validation CSV not found.")
  if (!file.exists(dev_csv)) stop("CRITICAL ERROR: Development CSV not found.")
  
  raw_dev <- read.csv(dev_csv, stringsAsFactors = FALSE)
  raw_val <- read.csv(val_csv, stringsAsFactors = FALSE)
  raw_data <- bind_rows(raw_dev, raw_val) %>% distinct()
  
} else {
  if (!file.exists(dev_csv)) stop("CRITICAL ERROR: Development CSV not found.")
  raw_data <- read.csv(dev_csv, stringsAsFactors = FALSE)
}

out_dir <- here("submission", dataset_label)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ------------------------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------------------------
raw_data <- raw_data %>%
  mutate(
    dt = parse_date_time(dt, orders = c("Ymd HMS", "Ymd HM", "Ymd", "dmy HMS", "dmy HM"), tz = "UTC"),
    date = as.Date(dt),
    time = format(dt, "%H:%M:%S")
  )

if (mode == "development") {
  cutoff <- as.POSIXct("2025-09-30 00:00:00", tz = "UTC")
  raw_data <- raw_data %>% filter(dt <= cutoff)
  message("MODO DEV: Datos truncados estrictamente hasta: ", cutoff)
} else {
  message("MODO VAL: Usando toda la historia disponible. Fecha máxima detectada: ", max(raw_data$dt, na.rm = TRUE))
}

forecasting_df <- raw_data %>%
  mutate(
    midday_day = if_else(format(dt, "%H:%M:%S") <= "12:00:00", date, date + 1)
  ) %>%
  select(-coverage, -coverage_label, -variable_type, -dt, -date, -time) %>%
  group_by(midday_day, metric_name) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    id_cols = midday_day,
    names_from = metric_name,
    values_from = value
  ) %>%
  arrange(midday_day)

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

forecasting_df <- forecasting_df %>%
  mutate(across(where(is.numeric), ~ ifelse(. < -100, NA, .)))

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
  
  tryCatch(
    imputeTS::na_kalman(x),
    error = function(e) {
      tryCatch(
        imputeTS::na_locf(x, option = "locf"),
        error = function(e) {
          med <- median(x, na.rm = TRUE)
          if (is.na(med)) med <- mean(x, na.rm = TRUE)
          if (is.na(med)) return(x)
          x[is.na(x)] <- med
          x
        }
      )
    }
  )
}

# ------------------------------------------------------------
# 3. Feature engineering & D-3 Helper (OPTIMIZED)
# ------------------------------------------------------------
forecasting_df <- forecasting_df %>%
  mutate(across(all_of(num_cols), impute_safe_ts)) %>%
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

# =====================================================================
# NEAR-ZERO VARIANCE CHECK
# =====================================================================
is_informative <- function(x) {
  if (is.na(sd(x, na.rm = TRUE)) || sd(x, na.rm = TRUE) < 1e-5) return(FALSE)
  val_counts <- table(x)
  if (length(val_counts) == 0) return(FALSE) 
  max_freq_ratio <- max(val_counts) / sum(!is.na(x))
  if (max_freq_ratio > 0.98) return(FALSE)
  return(TRUE)
}

valid_metrics <- forecasting_df %>%
  select(where(is.numeric)) %>%
  select(where(is_informative)) %>%
  names()

final_cols <- unique(c("midday_day", valid_metrics))
vars_removed <- ncol(forecasting_df) - length(final_cols)
forecasting_df <- forecasting_df %>% select(all_of(final_cols))

cat("\n--- Check ---\n")
cat("Variables dropped due to near-zero variance:", vars_removed, "\n\n")

# =====================================================================
# HIERARCHICAL CLUSTERING (BASE PREDICTORS)
# =====================================================================
message("Running Hierarchical Clustering to select unique base signals...")

exclude_from_cluster <- c(
  "midday_day", "estimated_avoidable_deaths",
  "dow", "dom", "month", "week", "is_weekend", 
  "sin_dow", "cos_dow", "sin_month", "cos_month"
)

base_numeric <- setdiff(names(forecasting_df)[sapply(forecasting_df, is.numeric)], exclude_from_cluster)

# Calculate correlation matrix
corr_base <- cor(forecasting_df[, base_numeric], use = "pairwise.complete.obs")
corr_base[is.na(corr_base)] <- 0 # Handle zero-variance edge cases post-filter

# Hierarchical clustering (cut at h=0.2 to keep variables < 0.8 correlated)
clusters <- hclust(as.dist(1 - abs(corr_base)), method = "average")
groups <- cutree(clusters, h = 0.2)

# Select 1 representative variable per cluster (highest correlation to target)
target_cor <- sapply(base_numeric, function(x) {
  abs(cor(forecasting_df[[x]], forecasting_df$estimated_avoidable_deaths, use = "complete.obs"))
})
target_cor[is.na(target_cor)] <- 0

selected_base_vars <- character()
for (g in unique(groups)) {
  cluster_vars <- names(groups[groups == g])
  best_var <- cluster_vars[which.max(target_cor[cluster_vars])]
  selected_base_vars <- c(selected_base_vars, best_var)
}

# Overwrite dataframe to keep only the optimal base features
keep_cols <- c(exclude_from_cluster, selected_base_vars)
forecasting_df <- forecasting_df[, names(forecasting_df) %in% keep_cols]

message(sprintf("Clustering reduced base predictors from %d to %d unique signals.", length(base_numeric), length(selected_base_vars)))

# =====================================================================
# ROLLING FEATURES & LAGS (STREAMLINED)
# =====================================================================
# Generate 7-day rolling mean
create_rolling_features <- function(data, vars) {
  result <- data
  for (var in vars) {
    result[[paste0(var, "_roll_mean_7")]] <-
      zoo::rollapply(data[[var]], width = 7, FUN = mean, fill = NA, align = "right", na.rm = TRUE)
  }
  result
}

base_predictors <- selected_base_vars
forecasting_df <- create_rolling_features(forecasting_df, base_predictors)

# Cleaned up target lags to avoid noise. Removed target roll_sd entirely.
forecasting_df <- forecasting_df %>%
  mutate(
    y_lag_3  = lag(estimated_avoidable_deaths, 3),
    y_lag_4  = lag(estimated_avoidable_deaths, 4),
    y_lag_7  = lag(estimated_avoidable_deaths, 7),
    y_lag_14 = lag(estimated_avoidable_deaths, 14),
    y_roll_mean_7  = lag(zoo::rollapply(estimated_avoidable_deaths, 7, mean, fill = NA, align = "right", na.rm = TRUE), 3),
    y_roll_mean_14 = lag(zoo::rollapply(estimated_avoidable_deaths, 14, mean, fill = NA, align = "right", na.rm = TRUE), 3)
  )

# Remove NA padding from the lag calculations
forecasting_df <- forecasting_df[-c(1:14), ]

target <- "estimated_avoidable_deaths"
horizon <- 10

# =====================================================================
# D-3 HELPER RULE
# =====================================================================
# Helper function updated to match the streamlined target variables
apply_d3_lag_rule <- function(test_df, orig_idx, df_full, target_var, h=10) {
  last_legal_idx <- orig_idx - 3
  last_legal_val <- if(last_legal_idx > 0) df_full[[target_var]][last_legal_idx] else NA
  
  legal_mean_7  <- df_full$y_roll_mean_7[orig_idx]
  legal_mean_14 <- df_full$y_roll_mean_14[orig_idx]
  
  for (step in 1:h) {
    if (((orig_idx + step) - 3) > last_legal_idx) {
      test_df$y_lag_3[step]  <- last_legal_val
      test_df$y_lag_4[step]  <- last_legal_val
      test_df$y_lag_7[step]  <- last_legal_val
      test_df$y_lag_14[step] <- last_legal_val
    }
    test_df$y_roll_mean_7[step]  <- legal_mean_7
    test_df$y_roll_mean_14[step] <- legal_mean_14
  }
  return(test_df)
}

# =====================================================================
# EXPLICIT TOP-TIER INTERACTIONS (XGBoost Shortcuts)
# =====================================================================
message("Adding targeted explicit interactions for top signals...")

# Safely add interactions only if the base variables survived the clustering filter
forecasting_df <- forecasting_df %>%
  mutate(
    # 1. Target trend stability (Recent vs Historical lag)
    interaction_y_mean7_lag7 = y_roll_mean_7 * y_lag_7,
    
    # 2. Recent trajectory ratio (Is the target accelerating or decelerating?)
    ratio_y_lag3_lag7 = if_else(y_lag_7 == 0 | is.na(y_lag_7), 0, y_lag_3 / y_lag_7),
    
    # 3. Main exogenous signal interactions (Checking if columns exist first)
    interaction_nofdtas_lag7 = if("nofdtas" %in% names(.)) nofdtas * y_lag_7 else NA,
    interaction_ahmlh1_lag3  = if("ahmlh.1" %in% names(.)) ahmlh.1 * y_lag_3 else NA,
    interaction_ambulncq_mean7 = if("ambulncq" %in% names(.)) ambulncq * y_roll_mean_7 else NA
  )

forecasting_df <- forecasting_df %>%
  mutate(across(starts_with("interaction_"), ~ ifelse(is.infinite(.), NA, .)))

message("Interactions added successfully.")

# ------------------------------------------------------------------------------
# 4. FIXED OPTIMAL HYPERPARAMETERS (Loaded dynamically from RDS)
# ------------------------------------------------------------------------------
message("Loading pre-tuned optimal hyperparameters from RDS for production...")

rds_filename <- "best_xgb_params.rds"

if (!file.exists(rds_filename)) {
  rds_filename <- file.path("submission", "best_xgb_params.rds")
}

if (file.exists(rds_filename)) {
  saved_params <- readRDS(rds_filename)
  
  # Model optimized for the short term (Days 1 to 5)
  best_params_1_5  <- as.list(saved_params$best_params_1_5)
  best_nrounds_1_5 <- as.numeric(saved_params$best_nrounds_1_5)
  
  # Model optimized for the medium term (Days 6 to 10)
  best_params_6_10  <- as.list(saved_params$best_params_6_10)
  best_nrounds_6_10 <- as.numeric(saved_params$best_nrounds_6_10)
  
  message("Success: Optimal hyperparameters successfully loaded from: ", rds_filename)
} else {
  stop("CRITICAL ERROR: 'best_xgb_params.rds' not found. Ensure the file is in the script folder.")
}

# ------------------------------------------------------------
# 5. Rolling-origin XGBoost
# ------------------------------------------------------------
fit_xgb_final_window <- function(train_data, test_data, target, params, opt_nrounds) {
  
  predictors <- setdiff(names(train_data), c("midday_day", target))
  
  keep_pred <- function(x) { is.numeric(x) && is.finite(sd(x, na.rm=TRUE)) && sd(x, na.rm=TRUE) > 0 }
  predictors <- predictors[sapply(train_data[, predictors, drop = FALSE], keep_pred)]
  
  X_train <- as.matrix(train_data[, predictors, drop = FALSE])
  y_train <- train_data[[target]]
  X_test  <- as.matrix(test_data[, predictors, drop = FALSE])
  
  valid_target_idx <- which(is.finite(y_train))
  
  X_train <- X_train[valid_target_idx, , drop = FALSE]
  y_train <- y_train[valid_target_idx]
  
  if(length(y_train) == 0) {
    stop("No training observations with valid target.")
  }
  
  X_train[!is.finite(X_train)] <- NA
  X_test[!is.finite(X_test)] <- NA
  
  dtrain <- xgb.DMatrix(X_train, label = y_train, missing = NA)
  dtest  <- xgb.DMatrix(X_test, missing = NA)
  
  model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = params$eta,
      max_depth = params$max_depth,
      min_child_weight = params$min_child_weight,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      gamma = params$gamma,
      lambda = params$lambda,
      alpha = params$alpha
    ),
    data = dtrain,
    nrounds = opt_nrounds,
    verbose = 0,
    maximize = FALSE
  )
  
  pred <- as.numeric(predict(model, dtest))
  
  list(pred_xgb = pred, n_features = as.integer(length(predictors)))
}

if (mode == "validation") {
  origins <- which(forecasting_df$midday_day >= as.Date("2025-09-30") & 
                     forecasting_df$midday_day <= as.Date("2026-02-07"))
} else {
  n <- nrow(forecasting_df)
  origins <- (n - horizon - 150):(n - horizon)
  origins <- origins[origins > 100]
}

origins <- origins[(origins + horizon) <= nrow(forecasting_df)]
n_forecasts <- length(origins)

pred_xgb_mat <- matrix(NA, nrow = n_forecasts, ncol = horizon)
actual_mat <- matrix(NA, nrow = n_forecasts, ncol = horizon)
window_runtime_sec <- numeric(n_forecasts)
n_features_used <- integer(n_forecasts)

for (i in seq_len(n_forecasts)) {
  start_i <- Sys.time()
  
  orig_idx <- origins[i]
  test_idx  <- (orig_idx + 1):(orig_idx + horizon)
  
  # Shield the test data with the D-3 Helper to prevent leakage
  test_data <- apply_d3_lag_rule(forecasting_df[test_idx, , drop = FALSE], orig_idx, forecasting_df, target, horizon)  
  
  # ------------------------------------------------------------
  # MODEL 1: Optimized for days 1 to 5
  # ------------------------------------------------------------
  hist_1_5 <- min(orig_idx, best_params_1_5$train_history)
  train_idx_1_5 <- (orig_idx - hist_1_5 + 1):orig_idx
  train_data_1_5 <- forecasting_df[train_idx_1_5, , drop = FALSE]
  fit_1_5 <- fit_xgb_final_window(train_data_1_5, test_data, target, best_params_1_5, best_nrounds_1_5)
  
  # ------------------------------------------------------------
  # MODEL 2: Optimized for days 6 to 10
  # ------------------------------------------------------------
  hist_6_10 <- min(orig_idx, best_params_6_10$train_history)
  train_idx_6_10 <- (orig_idx - hist_6_10 + 1):orig_idx
  train_data_6_10 <- forecasting_df[train_idx_6_10, , drop = FALSE]
  fit_6_10 <- fit_xgb_final_window(train_data_6_10, test_data, target, best_params_6_10, best_nrounds_6_10)
  
  # ------------------------------------------------------------
  # COMBINE PREDICTIONS
  # ------------------------------------------------------------
  # Use Model 1 predictions for the first 5 days, and Model 2 for the rest
  pred_final <- c(fit_1_5$pred_xgb[1:5], fit_6_10$pred_xgb[6:10])
  
  pred_xgb_mat[i, ] <- pred_final
  
  if (mode == "development") {
    actual_mat[i, ] <- test_data[[target]]
  }
  
  # Store how many predictors were used by model 1 (usually the same as model 2)
  n_features_used[i] <- fit_1_5$n_features
  
  window_runtime_sec[i] <- as.numeric(difftime(Sys.time(), start_i, units = "secs"))
  
  if (i %% 25 == 0 || i == n_forecasts) {
    message(sprintf("Completed %s / %s windows | Current progress stable.", i, n_forecasts))
  }
}

# ------------------------------------------------------------
# 6. Outputs & Reproducibility Summaries
# ------------------------------------------------------------
message("Generating reproducibility summaries for the report...")

# 6.1. Dataset summary
data_summary <- tibble(
  n_days = nrow(forecasting_df),
  n_predictors = ncol(forecasting_df) - 2, # Excluding midday_day and the target
  start_date = min(forecasting_df$midday_day),
  end_date = max(forecasting_df$midday_day)
)

# 6.2. Transformation summary (the missing piece!)
# Dynamically classifies your current columns for the report table
transformation_summary <- tibble(
  transformation = c("none", "rolling", "lags", "interactions"),
  n_predictors = c(
    sum(
      !grepl("_roll_|_lag_|interaction_", names(forecasting_df)) &
        !(names(forecasting_df) %in% c("midday_day", "estimated_avoidable_deaths"))
    ),
    sum(grepl("_roll_", names(forecasting_df))),
    sum(grepl("_lag_", names(forecasting_df))),
    sum(grepl("interaction_", names(forecasting_df)))
  )
)

# 6.3. Runtime performance summary
runtime_summary <- tibble(
  dataset            = dataset_label,
  train_window_days  = "Dynamic Hybrid (Tuned History)",
  horizon_days       = horizon,
  n_forecasts        = n_forecasts,
  total_runtime_min  = sum(window_runtime_sec) / 60,
  mean_runtime_sec   = mean(window_runtime_sec),
  median_runtime_sec = median(window_runtime_sec),
  mean_features_used = mean(n_features_used, na.rm = TRUE)
)

# 6.4. Prediction output matrix
pred_out <- as.data.frame(pred_xgb_mat)
colnames(pred_out) <- paste0("day_", 1:horizon)
pred_out$forecast_id <- seq_len(n_forecasts)
pred_out <- pred_out[, c("forecast_id", paste0("day_", 1:horizon))]

# 6.5. Validation window summary
validation_summary <- tibble(
  dataset = dataset_label,
  first_training_day = forecasting_df$midday_day[1],
  first_test_day     = forecasting_df$midday_day[origins[1] + 1],
  last_test_day      = forecasting_df$midday_day[max(origins) + horizon],
  n_forecasts        = n_forecasts
)

# 6.6. Run metadata
run_info <- tibble(
  mode          = mode,
  dataset_label = dataset_label,
  train_window  = "Dynamic Hybrid",
  horizon       = horizon,
  n_forecasts   = n_forecasts
)

# --- BASE SAVING (Applies to both modes) ---
write.csv(run_info,              file.path(out_dir, "run_info.csv"),              row.names = FALSE)
write.csv(data_summary,          file.path(out_dir, "data_summary.csv"),         row.names = FALSE)
write.csv(transformation_summary,file.path(out_dir, "transformation_summary.csv"), row.names = FALSE)
write.csv(runtime_summary,       file.path(out_dir, "runtime_summary.csv"),      row.names = FALSE)
write.csv(pred_out,              file.path(out_dir, "pred_matrix.csv"),          row.names = FALSE)
write.csv(validation_summary,    file.path(out_dir, "validation_summary.csv"),   row.names = FALSE)

# --- CONDITIONAL SAVING (Development only) ---
if (mode == "development") {
  
  # Mean squared error helper
  mse_vec <- function(a, p) mean((a - p)^2, na.rm = TRUE)
  
  # Aggregate MSE summary
  results_summary <- tibble(
    model    = paste0("Hybrid XGBoost (", dataset_label, ")"),
    mse_1_5  = mse_vec(actual_mat[, 1:5],  pred_xgb_mat[, 1:5]),
    mse_6_10 = mse_vec(actual_mat[, 6:10], pred_xgb_mat[, 6:10])
  )
  
  # MSE per forecast window
  mse_df <- tibble(
    forecast_id = seq_len(n_forecasts),
    mse_1_5     = rowMeans((actual_mat[, 1:5]   - pred_xgb_mat[, 1:5])^2,  na.rm = TRUE),
    mse_6_10    = rowMeans((actual_mat[, 6:10]  - pred_xgb_mat[, 6:10])^2, na.rm = TRUE)
  )
  
  write.csv(results_summary, file.path(out_dir, "results_summary.csv"), row.names = FALSE)
  write.csv(mse_df,          file.path(out_dir, "mse_summary.csv"),    row.names = FALSE)
  message("Error metrics calculated and saved (DEV mode).")
  
} else {
  message("Validation mode: Skipping MSE calculation (no ground truth available).")
}

message("All output files were created successfully in: ", out_dir)