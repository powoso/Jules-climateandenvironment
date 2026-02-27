import unittest
import pandas as pd
import numpy as np
from src.features.enso import process_oni
from src.features.temperature import process_temperature

class TestFeatures(unittest.TestCase):
    def test_process_oni(self):
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        df = pd.DataFrame({'ONI_Anomaly': np.random.randn(20)}, index=dates)

        df_proc = process_oni(df)

        # Check columns
        self.assertIn('ONI_Lag_3', df_proc.columns)
        self.assertIn('ENSO_Phase', df_proc.columns)
        self.assertIn('ONI_3M_Avg', df_proc.columns)

        # Check logic
        # If mean > 0.5, should be El Nino (or one of the encoded columns)
        # We need to check one row
        row = df_proc.iloc[10]
        if row['ONI_3M_Avg'] >= 0.5:
             self.assertEqual(row['ENSO_Phase'], 'El Nino')
        elif row['ONI_3M_Avg'] <= -0.5:
             self.assertEqual(row['ENSO_Phase'], 'La Nina')
        else:
             self.assertEqual(row['ENSO_Phase'], 'Neutral')

    def test_process_temperature(self):
        dates = pd.date_range('1900-01-01', periods=200, freq='MS')
        df = pd.DataFrame({'TempAnomaly': np.linspace(0, 1, 200)}, index=dates)

        df_proc = process_temperature(df)

        self.assertIn('Temp_MA_12M', df_proc.columns)
        self.assertIn('Temp_Trend_10Y_Diff', df_proc.columns)

        # Check Trend
        # 10 years = 120 months.
        # At index 130, value should be value[130] - value[10]
        val_diff = df_proc['Temp_Trend_10Y_Diff'].iloc[130]
        expected = df['TempAnomaly'].iloc[130] - df['TempAnomaly'].iloc[10]
        self.assertAlmostEqual(val_diff, expected)

if __name__ == '__main__':
    unittest.main()
