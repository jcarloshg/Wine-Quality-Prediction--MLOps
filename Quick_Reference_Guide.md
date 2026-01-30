# MLOps Quick Reference Guide

## 🚀 Quick Start Commands

### Initial Setup
```bash
# 1. Create project
mkdir wine-quality-mlops && cd wine-quality-mlops

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install mlflow scikit-learn pandas numpy matplotlib seaborn flask pytest

# 3. Create directories
mkdir -p data/{raw,processed} src models tests
```

### Running the Pipeline

```bash
# Start MLflow Server (Terminal 1)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# Run training (Terminal 2)
python src/train.py

# Register best model
python src/evaluate.py

# Start API server (Terminal 3)
python src/predict.py
```

---

## 📊 MLflow UI Navigation

### Access Points
- **Main UI**: http://localhost:5000
- **Experiments**: Click experiment name → See all runs
- **Compare**: Select runs → Compare
- **Models**: Click "Models" in top navigation

### Key Features
1. **Experiments Tab**: View all experiment runs
2. **Compare Runs**: Select multiple → Click "Compare"
3. **Model Registry**: Manage model versions
4. **Artifacts**: Download models, plots, data

---

## 🔧 MLflow Python API Essentials

### Tracking

```python
import mlflow

# Start tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

# Log a run
with mlflow.start_run(run_name="my-run"):
    # Log parameters
    mlflow.log_param("alpha", 0.5)
    mlflow.log_params({"l1": 0.1, "l2": 0.2})
    
    # Log metrics
    mlflow.log_metric("rmse", 0.85)
    mlflow.log_metrics({"mae": 0.6, "r2": 0.95})
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("plot.png")
```

### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
mlflow.register_model(
    model_uri="runs:/RUN_ID/model",
    name="my-model"
)

# Transition to production
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.sklearn.load_model("models:/my-model/Production")
```

### Search Runs

```python
# Search experiments
runs = mlflow.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.rmse < 0.8",
    order_by=["metrics.rmse ASC"],
    max_results=10
)

# Get best run
best_run = runs.iloc[0]
print(f"Best RMSE: {best_run['metrics.rmse']}")
```

---

## 🌐 API Testing

### Test Single Prediction

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### Python Client

```python
import requests

url = "http://localhost:5001/predict"
data = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    # ... other features
}

response = requests.post(url, json=data)
print(response.json())
```

### Batch Predictions

```bash
curl -X POST http://localhost:5001/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"fixed acidity": 7.4, "volatile acidity": 0.7, ...},
      {"fixed acidity": 8.1, "volatile acidity": 0.5, ...}
    ]
  }'
```

---

## 🐛 Common Issues & Solutions

### Issue: MLflow server won't start

```bash
# Check if port is in use
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
mlflow server --port 5050
```

### Issue: Import errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Model not found

```python
# List all registered models
from mlflow.tracking import MlflowClient
client = MlflowClient()
for model in client.search_registered_models():
    print(model.name)
```

### Issue: Database locked

```bash
# Stop all MLflow processes
pkill -f mlflow

# Delete lock file
rm mlflow.db-wal mlflow.db-shm

# Restart server
mlflow server ...
```

---

## 📈 Performance Tuning

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

with mlflow.start_run():
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Log best params
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_score", grid_search.best_score_)
```

### Automated Runs

```python
# Run multiple experiments automatically
params_list = [
    {'alpha': 0.1, 'l1_ratio': 0.5},
    {'alpha': 0.5, 'l1_ratio': 0.5},
    {'alpha': 1.0, 'l1_ratio': 0.5},
]

for params in params_list:
    with mlflow.start_run():
        model = ElasticNet(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
```

---

## 🔍 Monitoring Best Practices

### Log Predictions

```python
import json
from datetime import datetime

def log_prediction(features, prediction, actual=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': float(prediction),
        'actual': actual
    }
    
    with open('predictions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

### Calculate Metrics

```python
import pandas as pd

# Load logs
logs = pd.read_json('predictions.jsonl', lines=True)

# Calculate metrics
with_actuals = logs[logs['actual'].notna()]
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    with_actuals['actual'], 
    with_actuals['prediction']
)
print(f"Production MSE: {mse}")
```

---

## 📦 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t wine-quality-mlops -f docker/Dockerfile .

# Run container
docker run -p 5000:5000 -p 5001:5001 wine-quality-mlops

# Or use docker-compose
docker-compose -f docker/docker-compose.yml up
```

### Push to Registry

```bash
# Tag image
docker tag wine-quality-mlops:latest myregistry/wine-quality-mlops:v1

# Push
docker push myregistry/wine-quality-mlops:v1
```

---

## ✅ Pre-Deployment Checklist

- [ ] All experiments tracked in MLflow
- [ ] Best model registered and promoted to Production
- [ ] Model performance meets requirements (RMSE, R²)
- [ ] API endpoints tested and working
- [ ] Monitoring system in place
- [ ] Error handling implemented
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Docker image built and tested

---

## 🎓 MLOps Concepts Explained

### Experiment Tracking
**What**: Recording all details of model training runs  
**Why**: Reproducibility, comparison, debugging  
**How**: MLflow logs params, metrics, artifacts automatically

### Model Registry
**What**: Centralized repository for models  
**Why**: Version control, lifecycle management, governance  
**Stages**: None → Staging → Production → Archived

### Model Serving
**What**: Making models available for predictions  
**Why**: Enable applications to use trained models  
**Methods**: REST API, batch processing, streaming

### Model Monitoring
**What**: Tracking model performance in production  
**Why**: Detect drift, degradation, anomalies  
**Metrics**: Prediction distribution, accuracy, latency

---

## 📚 Learning Path

### Week 1: Basics
- ✅ Complete this wine quality project
- Read MLflow documentation
- Understand experiment tracking

### Week 2: Intermediate
- Add hyperparameter tuning
- Implement model comparison
- Build monitoring dashboard

### Week 3: Advanced
- Set up CI/CD pipeline
- Implement A/B testing
- Deploy to cloud (AWS/GCP/Azure)

### Week 4: Production
- Add data validation
- Implement feature store
- Set up alerting system

---

## 🔗 Useful Resources

- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Flask**: https://flask.palletsprojects.com/
- **Docker**: https://docs.docker.com/
- **MLOps Guide**: https://ml-ops.org/

---

## 💡 Pro Tips

1. **Always version your data** - Use DVC or similar
2. **Track everything** - When in doubt, log it
3. **Automate testing** - Catch issues early
4. **Document assumptions** - Help future you
5. **Monitor continuously** - Don't wait for failures
6. **Start simple** - Add complexity gradually
7. **Review experiments** - Learn from each run
8. **Use version control** - Git for code, MLflow for models

---

**Remember**: MLOps is about making ML systems reliable, reproducible, and maintainable. Take it step by step!
