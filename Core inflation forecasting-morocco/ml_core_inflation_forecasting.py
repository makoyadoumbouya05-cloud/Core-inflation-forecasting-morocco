"""
Machine Learning Models for Core Inflation Forecasting
Author: Makoya Doumbouya
Master's Thesis - Mohammed V University, Rabat

This script implements Random Forest and Ridge Regression models
to forecast Morocco's core inflation index (ISJ).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class InflationForecaster:
    """
    A class to handle feature engineering, model training, and forecasting
    for inflation time series data.
    """
    
    def __init__(self, data):
        """
        Initialize with ISJ inflation data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with Date index and ISJ column
        """
        self.df = data.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def engineer_features(self):
        """
        Create lagged features, rolling averages, and seasonal differences.
        """
        # Lagged values (1, 2, 3 months)
        self.df['ISJ_lag1'] = self.df['ISJ'].shift(1)
        self.df['ISJ_lag2'] = self.df['ISJ'].shift(2)
        self.df['ISJ_lag3'] = self.df['ISJ'].shift(3)
        
        # Rolling averages (short and medium-term trends)
        self.df['ISJ_roll3'] = self.df['ISJ'].rolling(window=3).mean()
        self.df['ISJ_roll6'] = self.df['ISJ'].rolling(window=6).mean()
        
        # 12-month seasonal difference
        self.df['ISJ_diff12'] = self.df['ISJ'].diff(12)
        
        # Remove rows with NaN created by transformations
        self.df = self.df.dropna()
        
        print(f"Features engineered. Dataset shape: {self.df.shape}")
        
    def prepare_data(self, test_size=0.2):
        """
        Split data into train/test sets (no shuffling for time series).
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        """
        feature_cols = ['ISJ_lag1', 'ISJ_lag2', 'ISJ_lag3', 
                       'ISJ_roll3', 'ISJ_roll6', 'ISJ_diff12']
        
        X = self.df[feature_cols]
        y = self.df['ISJ']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def train_random_forest(self, optimize=True):
        """
        Train Random Forest with optional hyperparameter tuning.
        
        Parameters:
        -----------
        optimize : bool
            Whether to perform GridSearchCV optimization
        
        Returns:
        --------
        model : RandomForestRegressor
            Trained model
        metrics : dict
            Performance metrics (RMSE, MAE, R²)
        """
        if optimize:
            print("Optimizing Random Forest hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            tscv = TimeSeriesSplit(n_splits=5)
            
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.X_train, self.y_train)
        
        # Predictions and metrics
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        return model, metrics
    
    def train_ridge(self, alpha=1.0):
        """
        Train Ridge Regression model.
        
        Parameters:
        -----------
        alpha : float
            Regularization strength
        
        Returns:
        --------
        model : Ridge
            Trained model
        metrics : dict
            Performance metrics (RMSE, MAE, R²)
        """
        print(f"Training Ridge Regression (alpha={alpha})...")
        model = Ridge(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        
        # Predictions and metrics
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        return model, metrics
    
    def forecast_ahead(self, model, n_months=12):
        """
        Generate iterative multi-step ahead forecasts.
        
        Parameters:
        -----------
        model : sklearn model
            Trained forecasting model
        n_months : int
            Number of months to forecast
        
        Returns:
        --------
        forecast_df : pd.DataFrame
            DataFrame with forecasts and confidence intervals
        """
        y_all = list(self.df['ISJ'])
        y_forecast = []
        
        for i in range(n_months):
            # Reconstruct features dynamically
            ISJ_lag1 = y_all[-1]
            ISJ_lag2 = y_all[-2] if len(y_all) > 1 else y_all[-1]
            ISJ_lag3 = y_all[-3] if len(y_all) > 2 else y_all[-1]
            ISJ_roll3 = np.mean(y_all[-3:]) if len(y_all) >= 3 else np.mean(y_all)
            ISJ_roll6 = np.mean(y_all[-6:]) if len(y_all) >= 6 else np.mean(y_all)
            ISJ_diff12 = y_all[-1] - y_all[-12] if len(y_all) >= 12 else 0
            
            current_X = pd.DataFrame(
                [[ISJ_lag1, ISJ_lag2, ISJ_lag3, ISJ_roll3, ISJ_roll6, ISJ_diff12]],
                columns=['ISJ_lag1', 'ISJ_lag2', 'ISJ_lag3', 
                        'ISJ_roll3', 'ISJ_roll6', 'ISJ_diff12']
            )
            
            next_pred = model.predict(current_X)[0]
            y_forecast.append(next_pred)
            y_all.append(next_pred)
        
        # Create future dates
        last_date = self.df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=n_months, 
            freq='MS'
        )
        
        # Simple confidence intervals (±1 std of residuals)
        residuals = self.y_test - model.predict(self.X_test)
        std_error = np.std(residuals)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': y_forecast,
            'Lower_CI': np.array(y_forecast) - 1.96 * std_error,
            'Upper_CI': np.array(y_forecast) + 1.96 * std_error
        })
        
        return forecast_df
    
    def plot_forecast(self, model, model_name='Model'):
        """
        Visualize historical data and forecasts.
        
        Parameters:
        -----------
        model : sklearn model
            Trained forecasting model
        model_name : str
            Name for plot title
        """
        forecast_df = self.forecast_ahead(model, n_months=12)
        
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(self.df.index, self.df['ISJ'], 
                label='Historical ISJ', color='black', linewidth=2)
        
        # Forecasts
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], 
                label=f'{model_name} Forecast', 
                linestyle='--', marker='o', color='red')
        
        # Confidence interval
        plt.fill_between(forecast_df['Date'], 
                        forecast_df['Lower_CI'], 
                        forecast_df['Upper_CI'],
                        color='orange', alpha=0.3, 
                        label='95% Confidence Interval')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Core Inflation Index (ISJ)', fontsize=12)
        plt.title(f'12-Month Inflation Forecast - {model_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate RMSE, MAE, and R²."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE:  {mae:.3f}")
        print(f"R²:   {r2:.3f}")
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Load data
    ipc_data = pd.DataFrame({
        'Date': pd.date_range(start='2017-01-01', periods=98, freq='MS'),
        'ISJ': [
            99.6, 99.9, 99.9, 99.7, 99.8, 99.9, 99.9, 99.9, 100.2, 100.4, 100.5, 100.5,
            100.8, 100.8, 100.8, 100.9, 101.1, 101.3, 101.3, 101.4, 101.9, 102.1, 102.2, 102.1,
            101.9, 101.8, 101.8, 101.7, 101.8, 101.7, 101.7, 102.0, 102.3, 102.3, 102.4, 102.4,
            102.5, 102.5, 102.5, 102.6, 102.6, 102.4, 102.4, 102.6, 102.5, 102.5, 102.6, 102.7,
            102.9, 103.1, 103.2, 103.5, 103.6, 103.7, 104.1, 104.3, 104.8, 105.3, 105.6, 105.9,
            106.2, 106.7, 107.2, 108.1, 109.4, 110.3, 110.9, 111.2, 112.1, 112.8, 113.6, 114.4,
            114.9, 115.8, 115.9, 116.3, 116.4, 116.5, 116.9, 116.7, 117.3, 117.7, 117.7, 117.9,
            118.2, 118.4, 118.7, 118.9, 119.0, 119.3, 119.4, 119.7, 120.1, 120.5, 120.8, 120.8,
            121.0, 121.2
        ]
    })
    ipc_data.set_index('Date', inplace=True)
    
    # Initialize forecaster
    forecaster = InflationForecaster(ipc_data)
    
    # Prepare data
    forecaster.engineer_features()
    forecaster.prepare_data(test_size=0.2)
    
    # Train Ridge Regression
    print("\n=== Ridge Regression ===")
    ridge_model, ridge_metrics = forecaster.train_ridge(alpha=1.0)
    
    # Train Random Forest
    print("\n=== Random Forest ===")
    rf_model, rf_metrics = forecaster.train_random_forest(optimize=False)
    
    # Visualize Ridge forecast
    forecaster.plot_forecast(ridge_model, model_name='Ridge Regression')