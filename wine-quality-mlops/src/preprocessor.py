# src/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from .config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR
)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def split_data(self, df):
        """Split data into train and test sets"""
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print(f"✅ Data split complete:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Standardize features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=self.feature_names,
            index=X_test.index
        )

        print("✅ Feature scaling complete")
        return X_train_scaled, X_test_scaled

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = PROCESSED_DATA_DIR / "train.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"✅ Processed data saved:")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")

    def save_scaler(self):
        """Save the fitted scaler"""
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")
        return scaler_path

    def load_scaler(self):
        """Load a saved scaler"""
        scaler_path = MODELS_DIR / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        return self.scaler


if __name__ == "__main__":
    from src.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_data()

    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_test)

    # Save
    preprocessor.save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test)
    preprocessor.save_scaler()
