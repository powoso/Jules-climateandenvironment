import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

class TemperatureModel:
    def __init__(self, use_arima=False):
        self.use_arima = use_arima
        self.model = None

    def train(self, df):
        """
        Trains the model.
        Args:
            df: DataFrame with 'TempAnomaly' as target and features.
        """
        if self.use_arima:
            # ARIMA uses TempAnomaly as endog and features as exog
            # We need to select relevant exog features.
            # We'll use ONI lags, trend.
            exog_cols = ['ONI_Lag_3', 'ONI_Lag_6', 'ONI_Lag_9', 'ONI_Lag_12', 'Temp_Trend_10Y_Diff']
            self.exog_cols = exog_cols

            endog = df['TempAnomaly']
            exog = df[exog_cols]

            # Use SARIMAX(1, 1, 1) as a simple baseline for time series
            self.model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
            self.model_fit = self.model.fit(disp=False)
            print(self.model_fit.summary())

        else:
            # Linear Regression Approach
            # Features: Lagged ONI, Month, Trend proxy
            # We'll use month dummies? Or cyclic encoding?
            # For simplicity, let's use Month number (1-12) directly or dummies if seasonality is strong.
            # But anomalies have seasonality removed usually.

            feature_cols = ['ONI_Lag_3', 'ONI_Lag_6', 'ONI_Lag_9', 'ONI_Lag_12', 'Temp_Trend_10Y_Diff']
            self.feature_cols = feature_cols

            X = df[feature_cols]
            y = df['TempAnomaly']

            self.model = LinearRegression()
            self.model.fit(X, y)

            # Print coefficients
            coef_df = pd.DataFrame({'Feature': feature_cols, 'Coef': self.model.coef_})
            print(coef_df)

    def predict(self, df):
        """
        Predicts temperature anomaly.
        """
        if self.use_arima:
            exog = df[self.exog_cols]
            # Predicting out of sample is tricky with statsmodels if indices don't align perfectly.
            # But for validation on holdout set:
            start = df.index[0]
            end = df.index[-1]
            pred = self.model_fit.predict(start=start, end=end, exog=exog)
            return pred
        else:
            X = df[self.feature_cols]
            return self.model.predict(X)

    def predict_next_year(self, last_row_df):
        """
        Predict next 12 months based on the last known data point.
        This is a bit complex as we need future values of lagged features.
        But for 1-step ahead it's easy. For 12-step ahead, we need to iterate.

        For this MVP, let's just predict the probability of exceeding a threshold for the NEXT month or year average
        assuming persistence of trend and known lags.
        """
        pass

def evaluate_model(df):
    # Split train/test
    # Let's hold out the last 2 years (24 months) for testing
    test_size = 24
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    print(f"Training on {len(train_df)} months, Testing on {len(test_df)} months.")

    # Train Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = TemperatureModel(use_arima=False)
    lr_model.train(train_df)

    preds_lr = lr_model.predict(test_df)
    rmse_lr = np.sqrt(mean_squared_error(test_df['TempAnomaly'], preds_lr))
    mae_lr = mean_absolute_error(test_df['TempAnomaly'], preds_lr)
    print(f"Linear Regression RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}")

    # Train ARIMA
    print("\n--- SARIMAX ---")
    arima_model = TemperatureModel(use_arima=True)
    arima_model.train(train_df)

    # For ARIMA, predict on test set
    # We need to use `get_prediction` or `forecast` with steps
    # But `predict(start, end)` works if dates are in index.
    preds_arima = arima_model.predict(test_df)

    rmse_arima = np.sqrt(mean_squared_error(test_df['TempAnomaly'], preds_arima))
    mae_arima = mean_absolute_error(test_df['TempAnomaly'], preds_arima)
    print(f"SARIMAX RMSE: {rmse_arima:.4f}, MAE: {mae_arima:.4f}")

    return lr_model, arima_model

if __name__ == "__main__":
    from src.features.pipeline import get_processed_data
    df = get_processed_data()
    # Ensure index is datetime and monotonic
    df = df.sort_index()
    evaluate_model(df)
