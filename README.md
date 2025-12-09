# Final Year Project — Customer Churn Prediction and CLTV Estimation with Explanable AI

This repository implements a simple pipeline to prepare data, train models, and score customers.

Project layout (important files and directories):
- [requirements.txt](requirements.txt)
- [docker-compose.yml](docker-compose.yml)
- [inputs/](inputs) (raw CSVs)
  - [inputs/customers.csv](inputs/customers.csv)
  - [inputs/transactions.csv](inputs/transactions.csv)
  - [inputs/events.csv](inputs/events.csv)
  - [inputs/support.csv](inputs/support.csv)
- [intermediate/](intermediate) (engineered outputs)
  - [intermediate/engineered_features.csv](intermediate/engineered_features.csv)
  - [intermediate/labels.csv](intermediate/labels.csv)
- [models/](models) (trained models)
  - [models/encoder.joblib](models/encoder.joblib)
  - [models/scaler.joblib](models/scaler.joblib)
  - [models/engineered_features.csv](models/engineered_features.csv)
  - [models/labels.csv](models/labels.csv)
- [outputs/](outputs) (final scored outputs)
  - [outputs/engineered_features.csv](outputs/engineered_features.csv)
  - [outputs/labels.csv](outputs/labels.csv)
- [src/](src) (source code)
  - [src/feature_engineering.py](src/feature_engineering.py)
  - [src/train.py](src/train.py)
  - [src/score.py](src/score.py)
  - [src/deploy_api.py](src/deploy_api.py)
- [deployment/](deployment) (alternate Docker configs)
  - [deployment/Dockerfile](deployment/Dockerfile)
  - [deployment/docker-compose.yml](deployment/docker-compose.yml)

Prerequisites
- Python 3.8+ installed
- Git (optional)
- Docker (optional, for container runs)

Local environment — quick start
1. Create and activate a virtual environment
   - macOS / Linux:
     
