import pandas as pd
import os
import logging
from src.collectors.downloader import download_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_gistemp(filepath):
    """
    Loads GISTEMP data from CSV.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None

    try:
        # Skip the first row which is just a title
        df = pd.read_csv(filepath, skiprows=1)
    except Exception as e:
        logging.error(f"Error reading GISTEMP file: {e}")
        return None

    # Replace '***' with NaN
    df = df.replace('***', float('nan'))

    # Convert columns to numeric
    # The file structure is Year, Jan, Feb...
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for col in months:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Melt to long format
    # We drop 'J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON' implicitly by only selecting Month columns
    id_vars = ['Year']
    value_vars = [m for m in months if m in df.columns]

    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Month', value_name='TempAnomaly')

    # Create Date column
    month_map = {m: i+1 for i, m in enumerate(months)}
    df_melted['MonthNum'] = df_melted['Month'].map(month_map)

    # Drop rows with missing Year or MonthNum
    df_melted = df_melted.dropna(subset=['Year', 'MonthNum'])

    try:
        df_melted['Date'] = pd.to_datetime(df_melted.apply(lambda x: f"{int(x['Year'])}-{int(x['MonthNum'])}-01", axis=1))
    except Exception as e:
        logging.error(f"Date conversion error: {e}")
        return None

    df_melted = df_melted.sort_values('Date').set_index('Date')
    return df_melted[['TempAnomaly']]

def load_oni(filepath):
    """
    Loads NOAA ONI data.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[1].isdigit():
             # SEAS YR TOTAL ANOM
             # Sometimes there might be extra spaces or columns, but typically the first 4 matter
            data.append(parts[:4])

    df = pd.DataFrame(data, columns=['SEAS', 'YR', 'TOTAL', 'ANOM'])
    df['YR'] = df['YR'].astype(int)
    df['TOTAL'] = df['TOTAL'].astype(float)
    df['ANOM'] = df['ANOM'].astype(float)

    seas_map = {
        'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
        'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
    }

    df['Month'] = df['SEAS'].map(seas_map)

    # NDJ is Dec, DJF is Jan.
    # We need to be careful. The file lists seasons.
    # DJF 1950 is Centered on Jan 1950.
    # NDJ 1950 is Centered on Dec 1950.
    # So Month is correct.

    try:
        df['Date'] = pd.to_datetime(df.apply(lambda x: f"{int(x['YR'])}-{int(x['Month'])}-01", axis=1))
    except Exception as e:
        logging.error(f"Date conversion error in ONI: {e}")
        return None

    df = df.sort_values('Date').set_index('Date')

    # Rename ANOM to ONI_Anomaly
    return df[['ANOM']].rename(columns={'ANOM': 'ONI_Anomaly'})

def get_merged_data(data_dir="data", download=True):
    """
    Downloads and loads merged data.
    """
    if download:
        download_data(data_dir)

    gistemp_path = os.path.join(data_dir, "GLB.Ts+dSST.csv")
    oni_path = os.path.join(data_dir, "oni.ascii.txt")

    df_temp = load_gistemp(gistemp_path)
    df_oni = load_oni(oni_path)

    if df_temp is None:
        raise FileNotFoundError("Could not load GISTEMP data.")
    if df_oni is None:
        raise FileNotFoundError("Could not load ONI data.")

    # Merge
    # We want to align on date.
    merged = df_temp.join(df_oni, how='inner')
    return merged

if __name__ == "__main__":
    try:
        df = get_merged_data()
        print("Merged Data Head:")
        print(df.head())
        print("\nMerged Data Tail:")
        print(df.tail())
    except Exception as e:
        print(f"Error: {e}")
