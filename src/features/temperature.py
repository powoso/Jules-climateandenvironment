import pandas as pd

def process_temperature(df):
    """
    Process Temperature data.

    Args:
        df: DataFrame with 'TempAnomaly' column, indexed by Date.

    Returns:
        DataFrame with added temperature features.
    """
    # Calculate moving averages
    df['Temp_MA_12M'] = df['TempAnomaly'].rolling(window=12).mean()
    df['Temp_MA_5Y'] = df['TempAnomaly'].rolling(window=60).mean()

    # Calculate trend (e.g. 10 year linear trend slope) - this is expensive in a rolling window.
    # Instead, we can use the difference from 10 years ago as a proxy for trend.
    df['Temp_Trend_10Y_Diff'] = df['TempAnomaly'] - df['TempAnomaly'].shift(120)

    # Month Feature (for seasonality, though anomalies shouldn't have much)
    df['Month'] = df.index.month

    return df
