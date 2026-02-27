import pandas as pd
from src.data_loader import get_merged_data
from src.features.enso import process_oni
from src.features.temperature import process_temperature

def get_processed_data(data_dir="data"):
    """
    Loads data and applies feature engineering.
    """
    # Load raw merged data (don't re-download to be safe)
    df = get_merged_data(data_dir, download=False)

    # Process ONI
    df = process_oni(df)

    # Process Temperature
    df = process_temperature(df)

    # Drop rows with NaN created by lags/rolling
    # We need to decide if we drop or keep.
    # For training, we drop.
    df_clean = df.dropna()

    return df_clean

if __name__ == "__main__":
    df = get_processed_data()
    print("Processed Data Head:")
    print(df.head())
    print("\nProcessed Data Columns:")
    print(df.columns)
