# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_STATE
)


class ModelTrainer:
    """Train and track ML models with MLflow"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        print(f"🔬 Experiment: {EXPERIMENT_NAME}")
        print(f"📍 Tracking URI: {MLFLOW_TRACKING_URI}")

    def load_processed_data(self):
        """Load preprocessed train and test data"""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

        X_train = train_df.drop(columns=['quality'])
        y_train = train_df['quality']
        X_test = test_df.drop(columns=['quality'])
        y_test = test_df['quality']

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        return metrics

    def plot_predictions(self, y_true, y_pred, model_name):
        """Create prediction plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Quality')
        axes[0].set_ylabel('Predicted Quality')
        axes[0].set_title(f'{model_name}: Predictions vs Actual')

        # Residuals
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residual Distribution')

        plt.tight_layout()

        # Save figure
        plot_path = f"/tmp/{model_name}_predictions.png"
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def train_model(self, model, model_name, params, X_train, y_train, X_test, y_test):
        """Train a single model with MLflow tracking"""

        with mlflow.start_run(run_name=model_name):
            print(f"\n🚀 Training {model_name}...")

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            train_metrics = self.evaluate_model(y_train, y_train_pred)
            test_metrics = self.evaluate_model(y_test, y_test_pred)

            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=None  # We'll register the best one later
            )

            # Create and log plots
            plot_path = self.plot_predictions(y_test, y_test_pred, model_name)
            mlflow.log_artifact(plot_path, "plots")

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                importance_path = f"/tmp/{model_name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")

            print(f"✅ {model_name} complete!")
            print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"   Test R²: {test_metrics['r2']:.4f}")

            return model, test_metrics

    def run_experiments(self):
        """Run multiple experiments with different models"""

        # Load data
        X_train, X_test, y_train, y_test = self.load_processed_data()

        # Define models to try
        models_config = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "random_state": RANDOM_STATE
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "random_state": RANDOM_STATE
                }
            },
            "ElasticNet": {
                "model": ElasticNet(random_state=RANDOM_STATE),
                "params": {
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": RANDOM_STATE
                }
            },
            "Ridge Regression": {
                "model": Ridge(random_state=RANDOM_STATE),
                "params": {
                    "alpha": 1.0,
                    "random_state": RANDOM_STATE
                }
            }
        }

        results = {}

        # Train each model
        for model_name, config in models_config.items():
            model = config["model"].set_params(**config["params"])
            _, metrics = self.train_model(
                model,
                model_name,
                config["params"],
                X_train,
                y_train,
                X_test,
                y_test
            )
            results[model_name] = metrics

        # Print summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        results_df = pd.DataFrame(results).T
        print(results_df.sort_values('rmse'))

        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   RMSE: {best_model[1]['rmse']:.4f}")
        print(f"   R²: {best_model[1]['r2']:.4f}")

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_experiments()
