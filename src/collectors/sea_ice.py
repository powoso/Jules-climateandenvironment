import pandas as pd
import numpy as np
import logging

def get_sea_ice_data(data_dir="data"):
    """
    Generates mock Arctic sea ice extent data for MVP.
    In a real system, this would download from NSIDC.
    """
    # Create a date range from 1979 to present
    dates = pd.date_range(start='1979-01-01', end='2025-12-31', freq='D')

    # Create a synthetic sea ice extent signal
    # Seasonal cycle: Max in March, Min in September
    # Trend: Declining

    day_of_year = dates.dayofyear

    # Seasonal component (approximate)
    # Extent ranges from ~15M km2 (March) to ~4M km2 (Sept)
    # Cosine wave
    seasonal = 10 + 5 * np.cos(2 * np.pi * (day_of_year - 80) / 365)

    # Trend component
    # -0.04 M km2 per year approx
    years_from_start = (dates.year - 1979) + (day_of_year / 365)
    trend = -0.08 * years_from_start

    # Random noise
    noise = np.random.normal(0, 0.5, size=len(dates))

    extent = seasonal + trend + noise

    # Ensure non-negative
    extent = np.maximum(extent, 0)

    df = pd.DataFrame({'Date': dates, 'Extent': extent})
    df = df.set_index('Date')

    return df

def predict_sea_ice_minimum(target_year):
    """
    Predicts the September minimum for the target year.
    """
    df = get_sea_ice_data()

    # Get historical minimums
    # Resample to annual minimum
    annual_min = df.resample('YE').min()

    # Simple linear regression on time
    # Fit on 1979-last_year
    X = annual_min.index.year.values.reshape(-1, 1)
    y = annual_min['Extent'].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    pred = model.predict([[target_year]])[0]

    # Calculate error standard deviation
    residuals = y - model.predict(X)
    std_dev = np.std(residuals)

    return pred, std_dev

if __name__ == "__main__":
    df = get_sea_ice_data()
    print(df.head())
    print(df.tail())

    year = 2026
    pred, std = predict_sea_ice_minimum(year)
    print(f"Predicted Minimum Extent for {year}: {pred:.2f} +/- {std:.2f} M km2")
