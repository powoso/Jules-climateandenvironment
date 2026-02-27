import unittest
from src.data_loader import get_merged_data
from src.features.pipeline import get_processed_data
from src.models.temperature_model import TemperatureModel

class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        # Should run without error
        try:
            df = get_processed_data(data_dir="data")
            self.assertIsNotNone(df)
            self.assertFalse(df.empty)
        except Exception as e:
            self.fail(f"Pipeline failed: {e}")

    def test_model_training(self):
        df = get_processed_data(data_dir="data")
        model = TemperatureModel(use_arima=True)
        try:
            model.train(df)
            self.assertIsNotNone(model.model_fit)
        except Exception as e:
            self.fail(f"Model training failed: {e}")

if __name__ == '__main__':
    unittest.main()
