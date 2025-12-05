# Final Year Project — Customer Churn Prediction and CLTV Estimation with explanable AI

This repository implements a simple pipeline to prepare data, train models, and score customers.

Project layout (important files):
- [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- inputs/ (raw CSVs)
  - [inputs/customers.csv](inputs/customers.csv)
  - [inputs/transactions.csv](inputs/transactions.csv)
  - [inputs/events.csv](inputs/events.csv)
  - [inputs/support.csv](inputs/support.csv)
- intermediate/ (engineered outputs)
  - [intermediate/engineered_features.csv](intermediate/engineered_features.csv)
  - [intermediate/labels.csv](intermediate/labels.csv)
- models/
  - [models/encoder.joblib](models/encoder.joblib)
  - [models/scaler.joblib](models/scaler.joblib)
  - [models/engineered_features.csv](models/engineered_features.csv)
  - [models/labels.csv](models/labels.csv)
- outputs/
  - [outputs/engineered_features.csv](outputs/engineered_features.csv)
  - [outputs/labels.csv](outputs/labels.csv)
- source
  - [`src.feature_engineering`](src/feature_engineering.py) — [src/feature_engineering.py](src/feature_engineering.py)
  - [`src.train`](src/train.py) — [src/train.py](src/train.py)
  - [`src.score`](src/score.py) — [src/score.py](src/score.py)
  - [`src.deploy_api:app`](src/deploy_api.py) — [src/deploy_api.py](src/deploy_api.py)
- deployment/ (alternate Docker configs)
  - [deployment/Dockerfile](deployment/Dockerfile)
  - [deployment/docker-compose.yml](deployment/docker-compose.yml)

Prerequisites
- Python 3.8+ installed
- Git (optional)
- Docker (optional, for container runs)

Local environment — quick start
1. Create and activate a virtual environment
   - macOS / Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

Run the pipeline (quick)
1. Place your input CSVs into [inputs/](inputs) — at minimum:
   - [inputs/customers.csv](inputs/customers.csv)
   - [inputs/transactions.csv](inputs/transactions.csv)
   - [inputs/events.csv](inputs/events.csv)
   - [inputs/support.csv](inputs/support.csv)

2. Feature engineering
   ```bash
   python src/feature_engineering.py --inputs inputs --output intermediate
   ```
   (See [`src.feature_engineering`](src/feature_engineering.py))

3. Train models
   ```bash
   python src/train.py --input intermediate --models models
   ```
   (See [`src.train`](src/train.py))

4. Score / produce outputs
   ```bash
   python src/score.py --intermediate intermediate --models models --output outputs
   ```
   (See [`src.score`](src/score.py))

Run the API (optional)
- Local dev server (Uvicorn):
  ```bash
  uvicorn src.deploy_api:app --reload --host 0.0.0.0 --port 8000
  ```
  See the FastAPI app at [`src.deploy_api:app`](src/deploy_api.py).

Docker (optional)
- Build and run locally with provided Dockerfile / compose:
  ```bash
  docker build -t customer-scoring .
  docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models customer-scoring
  ```
- Or use top-level compose:
  ```bash
  docker-compose up --build
  ```
- For deployment variants see [deployment/Dockerfile](deployment/Dockerfile) and [deployment/docker-compose.yml](deployment/docker-compose.yml).

Managing the project locally (best practices)
- Use a dedicated venv per branch or feature.
- Keep raw inputs in [inputs/](inputs); do not commit large CSVs to git.
- Store intermediate artifacts in [intermediate/] and final models in [models/].
- Pin dependencies in [requirements.txt](requirements.txt).
- Recreate environment when switching branches:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- For reproducible runs, run feature engineering -> train -> score in sequence.

Troubleshooting
- Missing inputs: ensure CSVs exist in [inputs/](inputs).
- Model artifacts absent: run training before scoring (see [`src.train`](src/train.py)).
- API errors: inspect Uvicorn output and the FastAPI app at [`src.deploy_api:app`](src/deploy_api.py).

License & notes
- This README assumes the CLI entrypoints are the scripts in [src/](src). Inspect those files for script-specific flags and options: [src/feature_engineering.py](src/feature_engineering.py), [src/train.py](src/train.py), [src/score.py](src/score.py), [src/deploy_api.py](src/deploy_api.py).