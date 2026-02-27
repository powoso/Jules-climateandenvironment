# Climate Prediction Markets System

A Python-based system for analyzing and predicting climate variables (Global Temperature Anomalies and Arctic Sea Ice Extent) to identify edges in prediction markets.

## Features

- **Data Collection**: Automatic downloading of NASA GISS Surface Temperature Analysis (GISTEMP v4) and NOAA Oceanic Niño Index (ONI) data.
- **Feature Engineering**: Processing of ENSO phases (El Niño/La Niña) and temperature trends.
- **Modeling**: SARIMAX time-series modeling to predict future global temperature anomalies based on lagged ENSO signals.
- **Edge Detection**: Calculation of probability distributions and comparison with market probabilities to identify betting edges (with Kelly Criterion sizing).
- **Sea Ice MVP**: A simple module for predicting Arctic Sea Ice Minimum extent.

## Installation on macOS

### Prerequisites

- Python 3.10 or higher
- `pip` (Python package installer)
- `git`

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/climate-prediction-system.git
    cd climate-prediction-system
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn scipy statsmodels
    ```

## Usage

The system is controlled via the `main.py` script.

### 1. Update Data
Download the latest data from NASA and NOAA.
```bash
python main.py update
```

### 2. Analyze Global Temperature Anomaly
Predict the probability of the global temperature anomaly exceeding a threshold for a specific year.
```bash
# Example: Will 2026 global temperature anomaly exceed 1.30°C?
# Market Probability: 40% (0.40)
python main.py temp --year 2026 --threshold 1.30 --market-prob 0.40
```

### 3. Analyze Arctic Sea Ice Minimum
Predict the minimum Arctic sea ice extent (in million km²) for a specific year.
```bash
# Example: Will Arctic Sea Ice Minimum be below 3.5 M km2 in 2026?
python main.py ice --year 2026 --threshold 3.5
```

## Project Structure

- `src/collectors`: Data downloading and generation scripts.
- `src/features`: Feature engineering logic.
- `src/models`: Predictive models (SARIMAX, Linear Regression).
- `src/analysis`: Market edge calculation logic.
- `data`: Directory where downloaded CSVs are stored.
- `tests`: Unit and integration tests.

## Running Tests

To verify the installation and code integrity:

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```
