# ============================================================================
# ARIMA Modeling for Morocco's Core Inflation (ISJ)
# Author: Makoya Doumbouya
# Master's Thesis - Mohammed V University
# ============================================================================

# Load required libraries
library(forecast)
library(tseries)
library(ggplot2)
library(gridExtra)
library(FinTS)
library(moments)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

ipc_data <- data.frame(
  Date = seq(as.Date("2017-01-01"), as.Date("2025-02-01"), by = "month"),
  ISJ = c(
    99.6, 99.9, 99.9, 99.7, 99.8, 99.9, 99.9, 99.9, 100.2, 100.4, 100.5, 100.5,
    100.8, 100.8, 100.8, 100.9, 101.1, 101.3, 101.3, 101.4, 101.9, 102.1, 102.2, 102.1,
    101.9, 101.8, 101.8, 101.7, 101.8, 101.7, 101.7, 102.0, 102.3, 102.3, 102.4, 102.4,
    102.5, 102.5, 102.5, 102.6, 102.6, 102.4, 102.4, 102.6, 102.5, 102.5, 102.6, 102.7,
    102.9, 103.1, 103.2, 103.5, 103.6, 103.7, 104.1, 104.3, 104.8, 105.3, 105.6, 105.9,
    106.2, 106.7, 107.2, 108.1, 109.4, 110.3, 110.9, 111.2, 112.1, 112.8, 113.6, 114.4,
    114.9, 115.8, 115.9, 116.3, 116.4, 116.5, 116.9, 116.7, 117.3, 117.7, 117.7, 117.9,
    118.2, 118.4, 118.7, 118.9, 119.0, 119.3, 119.4, 119.7, 120.1, 120.5, 120.8, 120.8,
    121.0, 121.2
  )
)

# ============================================================================
# 2. EXPLORATORY ANALYSIS
# ============================================================================

# Visualize the ISJ time series
png(ipc_data, widht=800, height=500)
ggplot(ipc_data, aes(x = Date, y = ISJ)) +
  geom_line(color = "black", linewidth = 1) +
  labs(title = "Monthly Evolution of Core Inflation Index (ISJ)",
       x = "Date",
       y = "ISJ Index") +
  theme_minimal()

# Descriptive statistics
cat("\n=== DESCRIPTIVE STATISTICS ===\n")
cat("Mean:", mean(ipc_data$ISJ), "\n")
cat("Median:", median(ipc_data$ISJ), "\n")
cat("SD:", sd(ipc_data$ISJ), "\n")
cat("Min:", min(ipc_data$ISJ), "\n")
cat("Max:", max(ipc_data$ISJ), "\n")
cat("Skewness:", skewness(ipc_data$ISJ), "\n")
cat("Kurtosis:", kurtosis(ipc_data$ISJ), "\n")

# Check for missing values
cat("\nMissing values:", sum(is.na(ipc_data$ISJ)), "\n")

# Boxplot for outlier detection
boxplot(ipc_data$ISJ,
        main = "Outlier Detection in ISJ Series",
        ylab = "Core Inflation Index (ISJ)",
        col = "lightblue",
        border = "darkblue")

# ============================================================================
# 3. STATIONARITY TESTING
# ============================================================================

# Convert to time series object
ipc_ts <- ts(ipc_data$ISJ, start = c(2017, 1), frequency = 12)

# Test stationarity on original series
cat("\n=== STATIONARITY TESTS: ORIGINAL SERIES ===\n")
adf_test <- adf.test(ipc_ts)
kpss_test <- kpss.test(ipc_ts)
print(adf_test)
print(kpss_test)

# First differencing
diff_ipc_ts <- diff(ipc_ts)

cat("\n=== STATIONARITY TESTS: FIRST DIFFERENCE ===\n")
adf_test_d1 <- adf.test(diff_ipc_ts)
kpss_test_d1 <- kpss.test(diff_ipc_ts)
print(adf_test_d1)
print(kpss_test_d1)

# Second differencing (needed for stationarity)
diff2_ipc_ts <- diff(ipc_ts, differences = 2)

cat("\n=== STATIONARITY TESTS: SECOND DIFFERENCE ===\n")
adf_test_d2 <- adf.test(diff2_ipc_ts)
kpss_test_d2 <- kpss.test(diff2_ipc_ts)
print(adf_test_d2)
print(kpss_test_d2)

# Visualize differenced series
plot(diff2_ipc_ts,
     main = "Second Differenced Series",
     ylab = "Differenced ISJ",
     xlab = "Time",
     col = "blue", lwd = 2)

# ============================================================================
# 4. ACF & PACF ANALYSIS
# ============================================================================

# ACF and PACF plots for model identification
acf_plot <- ggAcf(diff2_ipc_ts, lag.max = 20) + 
  ggtitle("ACF of Second Differenced Series")

pacf_plot <- ggPacf(diff2_ipc_ts, lag.max = 20) + 
  ggtitle("PACF of Second Differenced Series")

grid.arrange(acf_plot, pacf_plot, ncol = 2)

# ============================================================================
# 5. ARIMA MODEL ESTIMATION
# ============================================================================

# Manual model: ARIMA(1,2,1)
cat("\n=== MANUAL MODEL: ARIMA(1,2,1) ===\n")
model_manual <- Arima(ipc_ts, order = c(1, 2, 1))
summary(model_manual)

# Automatic model selection with auto.arima()
cat("\n=== AUTOMATIC MODEL SELECTION ===\n")
best_model <- auto.arima(ipc_ts, 
                         seasonal = FALSE, 
                         stepwise = FALSE, 
                         approximation = FALSE, 
                         trace = TRUE)
summary(best_model)

# ============================================================================
# 6. MODEL DIAGNOSTICS
# ============================================================================

cat("\n=== RESIDUAL DIAGNOSTICS ===\n")

# Check residuals
checkresiduals(best_model)

# Ljung-Box test (already included in checkresiduals output)

# ARCH-LM test for heteroscedasticity
cat("\n=== ARCH-LM TEST ===\n")
arch_test <- ArchTest(residuals(best_model), lags = 12)
print(arch_test)

# ============================================================================
# 7. FORECASTING
# ============================================================================

# Generate 12-month ahead forecasts
forecast_horizon <- 12
arima_forecast <- forecast(best_model, h = forecast_horizon)

# Display forecast summary
cat("\n=== FORECAST SUMMARY ===\n")
print(summary(arima_forecast))

# Visualize forecasts
plot(arima_forecast, 
     main = "12-Month Ahead Core Inflation Forecast", 
     ylab = "ISJ Index", 
     xlab = "Month")

# ============================================================================
# 8. MODEL PERFORMANCE EVALUATION
# ============================================================================

# Split data for validation
n <- length(diff2_ipc_ts)
train_size <- floor(0.8 * n)
train <- window(diff2_ipc_ts, start = start(diff2_ipc_ts), 
                end = time(diff2_ipc_ts)[train_size])
test <- window(diff2_ipc_ts, start = time(diff2_ipc_ts)[train_size + 1])

# Fit model on training data
model_train <- auto.arima(train, seasonal = FALSE)

# Forecast on test set
forecast_test <- forecast(model_train, h = length(test))

# Calculate performance metrics
actual_values <- as.numeric(test)
forecast_values <- forecast_test$mean

rmse <- sqrt(mean((forecast_values - actual_values)^2))
mae <- mean(abs(forecast_values - actual_values))
mape <- mean(abs((forecast_values - actual_values) / actual_values)) * 100
r2 <- 1 - sum((forecast_values - actual_values)^2) / 
  sum((actual_values - mean(actual_values))^2)

cat("\n=== PERFORMANCE METRICS (TEST SET) ===\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")
cat("R²:", r2, "\n")

cat("\n=== ANALYSIS COMPLETE ===\n")