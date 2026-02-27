import pandas as pd
import numpy as np

def process_oni(df):
    """
    Process ONI data to create ENSO features.

    Args:
        df: DataFrame with 'ONI_Anomaly' column, indexed by Date.

    Returns:
        DataFrame with added ENSO features.
    """
    # Create lagged features
    lags = [1, 3, 6, 9, 12]
    for lag in lags:
        df[f'ONI_Lag_{lag}'] = df['ONI_Anomaly'].shift(lag)

    # Moving averages of ONI
    df['ONI_3M_Avg'] = df['ONI_Anomaly'].rolling(window=3).mean()

    # ENSO Phase
    # El Nino: >= 0.5
    # La Nina: <= -0.5
    # Neutral: Otherwise
    # This is a simplification. NOAA definition requires 5 consecutive overlapping seasons.
    # But for feature engineering, the instantaneous or short-term average value is often used.

    conditions = [
        (df['ONI_3M_Avg'] >= 0.5),
        (df['ONI_3M_Avg'] <= -0.5)
    ]
    choices = ['El Nino', 'La Nina']
    df['ENSO_Phase'] = np.select(conditions, choices, default='Neutral')

    # One-hot encode ENSO Phase
    # We use pd.get_dummies but we need to join it back
    dummies = pd.get_dummies(df['ENSO_Phase'], prefix='ENSO')
    df = df.join(dummies)

    return df
