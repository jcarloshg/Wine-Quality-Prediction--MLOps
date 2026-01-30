# src/evaluate.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)

class ModelRegistry:
    """Manage model versioning and promotion"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    def get_best_run(self, metric="test_rmse"):
        """Find the best run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )

        if runs.empty:
            print("❌ No runs found")
            return None

        best_run = runs.iloc[0]
        print(f"\n🏆 Best Run Found:")
        print(f"   Run ID: {best_run.run_id}")
        print(f"   Model: {best_run['tags.mlflow.runName']}")
        print(f"   Test RMSE: {best_run['metrics.test_rmse']:.4f}")
        print(f"   Test R²: {best_run['metrics.test_r2']:.4f}")

        return best_run

    def register_best_model(self, metric="test_rmse"):
        """Register the best model to the Model Registry"""

        best_run = self.get_best_run(metric)

        if best_run is None:
            return None

        # Register model
        model_uri = f"runs:/{best_run.run_id}/model"

        print(f"\n📝 Registering model to '{REGISTERED_MODEL_NAME}'...")

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME
        )

        print(f"✅ Model registered:")
        print(f"   Name: {REGISTERED_MODEL_NAME}")
        print(f"   Version: {model_version.version}")

        return model_version

    def promote_model_to_production(self, version=None):
        """Promote a model version to Production stage"""

        if version is None:
            # Get latest version
            latest_versions = self.client.get_latest_versions(
                REGISTERED_MODEL_NAME,
                stages=["None"]
            )
            if not latest_versions:
                print("❌ No model versions found")
                return None
            version = latest_versions[0].version

        print(f"\n🚀 Promoting model version {version} to Production...")

        # Transition to production
        self.client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"✅ Model version {version} is now in Production!")

        return version

    def list_registered_models(self):
        """List all registered models and their versions"""
        try:
            models = self.client.search_registered_models()

            if not models:
                print("📭 No registered models found")
                return

            print("\n📚 Registered Models:")
            print("="*60)

            for model in models:
                print(f"\nModel: {model.name}")
                for version in model.latest_versions:
                    print(f"  Version {version.version}: {version.current_stage}")

        except Exception as e:
            print(f"❌ Error listing models: {e}")

    def load_production_model(self):
        """Load the production model"""
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"

        print(f"\n📦 Loading production model...")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully")

        return model

if __name__ == "__main__":
    registry = ModelRegistry()

    # Register best model
    model_version = registry.register_best_model()

    # Promote to production
    if model_version:
        registry.promote_model_to_production(model_version.version)

    # List all models
    registry.list_registered_models()