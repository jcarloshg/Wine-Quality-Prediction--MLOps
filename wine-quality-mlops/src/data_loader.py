# src/data_loader.py
import pandas as pd
import urllib.request
from pathlib import Path
from src.config import DATASET_URL, RAW_DATA_DIR, TARGET_COLUMN


class DataLoader:
    """Handle data downloading and loading"""

    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR / "winequality-red.csv"

    def download_data(self):
        """Download wine quality dataset"""
        if self.raw_data_path.exists():
            print(f"✅ Dataset already exists at {self.raw_data_path}")
            return self.raw_data_path

        print(f"📥 Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, self.raw_data_path)
        print(f"✅ Dataset downloaded to {self.raw_data_path}")
        return self.raw_data_path

    def load_data(self):
        """Load dataset into pandas DataFrame"""
        if not self.raw_data_path.exists():
            self.download_data()

        df = pd.read_csv(self.raw_data_path, sep=';')
        print(f"\n📊 Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {df.columns.tolist()}")
        print(
            f"   Target range: {df[TARGET_COLUMN].min()} - {df[TARGET_COLUMN].max()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")

        return df

    def validate_data(self, df):
        """Basic data validation"""
        assert not df.empty, "Dataset is empty"
        assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' not found"
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
        print("✅ Data validation passed")
        return True


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    loader.validate_data(df)
    print(f"\n{df.head()}")
