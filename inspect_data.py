import pandas as pd
import io

def load_gistemp(filepath):
    # Skip the first row which is just a title
    df = pd.read_csv(filepath, skiprows=1)
    # The file has Year, Jan, Feb, ... Dec, J-D, D-N, DJF, MAM, JJA, SON
    # We only want monthly data.
    # Replace '***' with NaN
    df = df.replace('***', float('nan'))

    # Convert columns to numeric
    cols_to_convert = df.columns[1:]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Melt to long format
    df_melted = df.melt(id_vars=['Year'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], var_name='Month', value_name='TempAnomaly')

    # Create Date column
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df_melted['MonthNum'] = df_melted['Month'].map(month_map)
    df_melted['Date'] = pd.to_datetime(df_melted[['Year', 'MonthNum']].rename(columns={'Year': 'year', 'MonthNum': 'month'}).assign(day=1))

    df_melted = df_melted.sort_values('Date').set_index('Date')
    return df_melted[['TempAnomaly']]

def load_oni(filepath):
    # The file is fixed width or space separated.
    # It has headers: SEAS  YR   TOTAL   ANOM
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit():
            data.append(parts)

    df = pd.DataFrame(data, columns=['SEAS', 'YR', 'TOTAL', 'ANOM'])
    df['YR'] = df['YR'].astype(int)
    df['TOTAL'] = df['TOTAL'].astype(float)
    df['ANOM'] = df['ANOM'].astype(float)

    # Convert Season to Month. DJF -> Jan, JFM -> Feb, etc.
    # Actually ONI is a 3 month running mean. DJF is centered on January.
    # Let's verify this mapping.
    # DJF -> Jan (1)
    # JFM -> Feb (2)
    # ...
    # NDJ -> Dec (12) of the PREVIOUS year? Or is 'YR' the year of the central month?
    # Let's look at the file content again.
    # DJF 1950.
    # Usually DJF 1950 means Dec 1949, Jan 1950, Feb 1950. Centered on Jan 1950.
    # Let's assume the Year column corresponds to the central month.

    seas_map = {
        'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
        'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
    }

    df['Month'] = df['SEAS'].map(seas_map)

    # Note: NDJ is centered on Dec. The year in the file:
    # NDJ 1950. This likely means Nov 1950, Dec 1950, Jan 1951. Centered on Dec 1950.
    # Let's check consistency.
    # DJF 1950 -> Jan 1950.
    # So if Month is 12 (NDJ), it is Dec of that year.

    df['Date'] = pd.to_datetime(df[['YR', 'Month']].rename(columns={'YR': 'year', 'Month': 'month'}).assign(day=1))
    df = df.sort_values('Date').set_index('Date')
    return df[['ANOM']]

if __name__ == "__main__":
    df_temp = load_gistemp('data/GLB.Ts+dSST.csv')
    print("GISTEMP head:")
    print(df_temp.head())

    df_oni = load_oni('data/oni.ascii.txt')
    print("\nONI head:")
    print(df_oni.head())

    # Merge
    merged = df_temp.join(df_oni, how='inner', rsuffix='_ONI')
    print("\nMerged head:")
    print(merged.head())
