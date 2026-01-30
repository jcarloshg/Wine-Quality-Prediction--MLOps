# src/monitor.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


class ModelMonitor:
    """Monitor model performance and predictions"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.predictions_log = Path("predictions_log.json")

    def log_prediction(self, features, prediction, actual=None):
        """Log a prediction for monitoring"""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }

        # Append to log file
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def load_prediction_logs(self):
        """Load all prediction logs"""

        if not self.predictions_log.exists():
            return pd.DataFrame()

        logs = []
        with open(self.predictions_log, 'r') as f:
            for line in f:
                logs.append(json.loads(line))

        return pd.DataFrame(logs)

    def calculate_drift(self, recent_window=100):
        """Calculate prediction drift"""

        logs_df = self.load_prediction_logs()

        if logs_df.empty or len(logs_df) < recent_window:
            print("⚠️  Not enough data for drift detection")
            return None

        # Get recent predictions
        recent_preds = logs_df.tail(recent_window)['prediction']

        # Calculate statistics
        stats = {
            'mean': recent_preds.mean(),
            'std': recent_preds.std(),
            'min': recent_preds.min(),
            'max': recent_preds.max(),
            'count': len(recent_preds)
        }

        print(f"\n📊 Recent Predictions Statistics (last {recent_window}):")
        for key, value in stats.items():
            print(f"   {key}: {value:.4f}")

        return stats

    def calculate_performance_metrics(self):
        """Calculate model performance on logged predictions with actuals"""

        logs_df = self.load_prediction_logs()

        # Filter for entries with actual values
        with_actuals = logs_df[logs_df['actual'].notna()]

        if with_actuals.empty:
            print("⚠️  No actual values logged for performance calculation")
            return None

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = with_actuals['actual'].values
        y_pred = with_actuals['prediction'].values

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'samples': len(with_actuals)
        }

        print(f"\n📈 Production Performance Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

        return metrics

    def get_model_metrics_history(self):
        """Get historical metrics from all runs"""

        from src.config import EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print("❌ No runs found")
            return None

        # Select relevant columns
        metrics_cols = [
            col for col in runs.columns if col.startswith('metrics.test_')]
        display_cols = ['run_id', 'tags.mlflow.runName',
                        'start_time'] + metrics_cols

        print(f"\n📊 Historical Model Performance:")
        print(runs[display_cols].head(10))

        return runs[display_cols]


if __name__ == "__main__":
    monitor = ModelMonitor()

    # Example: Log some predictions
    print("📝 Logging example predictions...")

    example_features = {
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

    # Simulate predictions with actual values
    for i in range(10):
        prediction = 5.0 + np.random.randn() * 0.5
        actual = 5 + np.random.choice([-1, 0, 1])
        monitor.log_prediction(example_features, prediction, actual)

    # Calculate metrics
    monitor.calculate_drift()
    monitor.calculate_performance_metrics()
    monitor.get_model_metrics_history()
