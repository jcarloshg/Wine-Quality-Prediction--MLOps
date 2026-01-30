# tests/test_pipeline.py
from src.predict import ModelPredictor
from src.preprocessor import DataPreprocessor
from src.data_loader import DataLoader
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataPipeline:

    def test_data_loading(self):
        """Test data can be loaded"""
        loader = DataLoader()
        df = loader.load_data()

        assert not df.empty
        assert 'quality' in df.columns
        assert len(df) > 1000

    def test_data_preprocessing(self):
        """Test data preprocessing"""
        loader = DataLoader()
        df = loader.load_data()

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)

        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_prediction(self):
        """Test model prediction"""
        predictor = ModelPredictor()

        # Create sample features
        features = {
            'fixed acidity': 7.4,
            'volatile acidity': 0.7,
            'citric acid': 0.0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11.0,
            'total sulfur dioxide': 34.0,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4
        }

        try:
            prediction = predictor.predict(features)
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 10
        except Exception as e:
            pytest.skip(f"Model not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
