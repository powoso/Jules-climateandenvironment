import pandas as pd
import numpy as np
from src.models.temperature_model import TemperatureModel
from src.features.pipeline import get_processed_data

def calculate_edge(market_probability, model_probability):
    """
    Calculates the edge and expected value.
    """
    edge = model_probability - market_probability

    if market_probability <= 0 or market_probability >= 1:
        return edge, 0.0

    b = (1 - market_probability) / market_probability
    p = model_probability
    q = 1 - p

    kelly = (b * p - q) / b
    kelly = max(0.0, kelly)

    return edge, kelly

def analyze_market_question(target_year, threshold, market_prob=None):
    """
    Analyzes a market question: "Will global temp anomaly exceed {threshold} in {target_year}?"
    """
    df = get_processed_data()

    # We use the SARIMAX model.
    model = TemperatureModel(use_arima=True)
    model.train(df)

    last_date = df.index[-1]
    last_year = last_date.year

    if target_year <= last_year:
        # Check historical data
        hist_data = df[df.index.year == target_year]['TempAnomaly']
        actual_avg = hist_data.mean()
        print(f"Historical Actual for {target_year}: {actual_avg:.2f} C")
        model_prob = 1.0 if actual_avg > threshold else 0.0
        if market_prob is not None:
             edge, kelly = calculate_edge(market_prob, model_prob)
             print(f"Model Prob: {model_prob:.1%}, Market: {market_prob:.1%}, Edge: {edge:.1%}")
        return model_prob, 0.0

    # Forecasting
    # We need to forecast enough months to cover the target year.
    # e.g. if last date is Dec 2025, and target is 2026, we need 12 months.

    months_to_forecast = 12
    if target_year > last_year + 1:
        # Simple logic: we only support next year forecast for now in this MVP
        print(f"Target year {target_year} is too far.")
        return None, None

    steps = months_to_forecast

    # Create future exog
    # Repeat last known values for ONI and Trend
    future_exog = pd.DataFrame(index=pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS'))
    last_row = df.iloc[-1]

    for col in model.exog_cols:
        future_exog[col] = last_row[col]

    # Statsmodels simulate() with exog requires exog to be (nsimulations, k_exog) if repetitions is NOT used?
    # No, usually simulate(nsimulations=steps, exog=future_exog, repetitions=1000)
    # The error was "Required (1000, 5), got (12, 5)".
    # This suggests nsimulations was interpreted as 1000 steps?
    # Ah, simulate(nsimulations=...) is the number of steps to simulate.
    # I passed nsimulations=1000. It should be nsimulations=steps.
    # And repetitions=1000.

    simulations = model.model_fit.simulate(nsimulations=steps, anchor='end', exog=future_exog, repetitions=1000)
    # simulations shape: (steps, repetitions)

    # simulations is a DataFrame with shape (steps, repetitions) if repetitions > 1?
    # Actually simulate returns a ndarray if repetitions > 1? Or a DataFrame?
    # Let's check type.

    sim_values = simulations
    if isinstance(sim_values, pd.DataFrame):
        sim_values = sim_values.values

    # Shape check
    # If repetitions=1000, it should be (steps, 1000)

    # Calculate annual average for each simulation path
    # Axis 0 is time (steps), Axis 1 is repetition
    annual_avgs = np.mean(sim_values, axis=0)

    # Probability
    prob_exceed = np.mean(annual_avgs > threshold)

    print(f"Model Probability for {target_year} > {threshold}C: {prob_exceed:.1%}")

    edge = 0.0
    if market_prob is not None:
        edge, kelly = calculate_edge(market_prob, prob_exceed)
        print(f"Market Probability: {market_prob:.1%}")
        print(f"Edge: {edge:.1%}")
        print(f"Kelly Fraction: {kelly:.2f}")

    return prob_exceed, edge

if __name__ == "__main__":
    print("Analyzing 2026 > 1.30C")
    analyze_market_question(2026, 1.30, market_prob=0.40)

    print("\nAnalyzing 2026 > 1.50C")
    analyze_market_question(2026, 1.50, market_prob=0.20)
