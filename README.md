# Customer Churn Prediction and CLTV Estimation with Explainable AI

This repository contains the source code for a final year project focused on predicting customer churn and estimating Customer Lifetime Value (CLTV). The project implements a machine learning pipeline to process data, train predictive models, and serve the results via an API with an interactive web-based frontend.

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started (Local Development)](#getting-started-local-development)
- [Usage (Running the Pipeline Locally)](#usage-running-the-pipeline-locally)
- [Running the Frontend](#running-the-frontend)
- [Complete Application Setup](#complete-application-setup)
- [Docker & Deployment](#docker--deployment)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Structure
The repository is organized to separate data, source code, models, and outputs for clarity and reproducibility.

```
project_code/
├── inputs/                      # Raw input data (CSV files)
├── intermediate/                # Intermediate processed data
├── models/                       # Trained model artifacts
│   ├── churn_model.pkl          # Churn classifier
│   ├── cltv_model.pkl           # CLTV regressor
│   ├── scaler.joblib            # Feature scaler
│   └── encoder.joblib           # Label encoder for plan types
├── outputs/                      # Final outputs (scored customers)
├── src/                          # Core Python source code
│   ├── feature_engineering.py    # Data preprocessing & feature creation
│   ├── train.py                  # Model training script
│   ├── score.py                  # Customer scoring script
│   └── deploy_api.py             # FastAPI server for predictions
├── templates/                    # HTML frontend
│   └── index.html                # Interactive web interface
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
└── requirements.txt              # Python dependencies
```

## Prerequisites
Before you begin, ensure you have the following installed:
- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Docker** (optional, for containerized deployment)
- **Docker Compose** (optional, for multi-container orchestration)
- **Git** (optional, for cloning the repository)

## Getting Started (Local Development)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project_code
```

### 2. Create and Activate a Virtual Environment
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```bash
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

## Usage (Running the Pipeline Locally)
The project is divided into several steps. Run them in the following order:

### Step 1: Run Feature Engineering
This script processes the raw data from the `inputs/` directory and saves the engineered features to the `intermediate/` directory.
```bash
python src/feature_engineering.py
```
**Output:** Engineered features saved to `intermediate/` directory

### Step 2: Train the Models
This script uses the engineered features to train the churn and CLTV models, saving the artifacts to the `models/` directory.
```bash
python src/train.py
```
**Output:** 
- `models/churn_model.pkl` - Trained churn classifier
- `models/cltv_model.pkl` - Trained CLTV regressor
- `models/scaler.joblib` - Feature scaler
- `models/encoder.joblib` - Plan type encoder

### Step 3: Score the Customers
This script applies the trained models to generate scores for customers.
```bash
python src/score.py
```
**Output:** Scored customer data in `outputs/` directory

### Step 4: Run the API Server
To serve the model's predictions via a REST API, run the following command. The API will be accessible at `http://127.0.0.1:8000`.
```bash
python src/deploy_api.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup - loading models...
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

## Running the Frontend

### Prerequisites for Frontend
The frontend is a modern web application that communicates with the FastAPI backend. No additional installations are required beyond the Python dependencies.

### Running Frontend with Backend

#### Option 1: Using Python's Built-in HTTP Server (Recommended for Development)

**Step 1:** Start the FastAPI backend in one terminal:
```bash
python src/deploy_api.py
```

**Step 2:** In another terminal, start a simple HTTP server to serve the frontend:
```bash
# From the project root directory
cd templates
python -m http.server 8080
```

**Step 3:** Open your browser and navigate to:
```
http://localhost:8080
```

#### Option 2: Using FastAPI StaticFiles (Advanced)

Modify `deploy_api.py` to serve static files:

```python
from fastapi.staticfiles import StaticFiles

# Add this after creating the FastAPI app
app.mount("/", StaticFiles(directory="templates", html=True), name="static")
```

Then run:
```bash
python src/deploy_api.py
```

Access the application at: `http://localhost:8000`

### Frontend Features

The interactive web interface includes:

✨ **Features:**
- **Customer Input Form** - Easy-to-use form for entering customer data
- **Real-time Predictions** - Get instant churn risk and CLTV predictions
- **Risk Assessment** - Visual risk indicators (High/Medium/Low)
- **Data Visualization** - Multiple charts:
  - 📊 Churn Risk Gauge (Doughnut Chart)
  - 💰 CLTV Distribution (Bar Chart)
  - 📈 Feature Impact Analysis (Horizontal Bar Chart)
  - 🎯 Customer Engagement Score (Radar Chart)
- **Input Summary** - Review all submitted customer data
- **Export Options** - Download reports and print results
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Dark/Light Theme Support** - Modern gradient UI

### API Endpoints

#### GET `/health`
Health check endpoint to verify API status.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": {
    "churn_model": true,
    "cltv_model": true,
    "scaler": true,
    "encoder": true
  },
  "model_dir": "models"
}
```

#### POST `/predict`
Make predictions for a customer.

**Request Body:**
```json
{
  "monthly_fee": 50.0,
  "total_amount": 500.0,
  "trans_count": 10,
  "recency_days": 30.0,
  "recency_days_tx": 10.0,
  "tenure_days": 365.0,
  "logins_7d": 5,
  "logins_30d": 20,
  "logins_90d": 60,
  "tickets": 2,
  "avg_resolution": 24.0,
  "plan": "standard"
}
```

**Response:**
```json
{
  "churn_prob": 0.25,
  "cltv_pred": 2500.50,
  "explanation": {
    "churn_model": {
      "type": "LogisticRegression"
    },
    "cltv_model": {
      "type": "LinearRegression"
    }
  }
}
```

## Complete Application Setup

### Running Everything Together (Manual)

**Terminal 1 - Backend API:**
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.\.venv\Scripts\activate  # Windows

# Start the API
python src/deploy_api.py
```

**Terminal 2 - Frontend Server:**
```bash
cd templates
python -m http.server 8080
```

**Terminal 3 (Optional) - Monitor Logs:**
```bash
tail -f deploy_api.log  # macOS/Linux
# OR
Get-Content -Path deploy_api.log -Wait  # Windows PowerShell
```

Then open your browser:
```
http://localhost:8080
```

## Docker & Deployment

### Running with Docker Compose (Recommended for Production)

Docker Compose manages both the API and frontend services automatically.

#### 1. Build and Run
From the project root directory:
```bash
docker-compose up --build
```

To run in background mode:
```bash
docker-compose up --build -d
```

#### 2. Access the Application
```
Frontend: http://localhost:80
API: http://localhost:8000
```

#### 3. Stop Services
```bash
docker-compose down
```

#### 4. View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f frontend
```

### Docker Compose Configuration

The `docker-compose.yml` includes:
- **api** service: FastAPI backend on port 8000
- **frontend** service: Nginx web server on port 80
- Volume mounts for model persistence
- Environment variable configuration

### Building Docker Image Manually

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-prediction-api
```

## API Documentation

### Interactive API Documentation
Once the API is running, visit:
```
http://localhost:8000/docs
```

This provides an interactive Swagger UI where you can test endpoints directly.

### Example API Call with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 50,
    "total_amount": 500,
    "trans_count": 10,
    "recency_days": 30,
    "recency_days_tx": 10,
    "tenure_days": 365,
    "logins_7d": 5,
    "logins_30d": 20,
    "logins_90d": 60,
    "tickets": 2,
    "avg_resolution": 24,
    "plan": "standard"
  }'
```

### Example API Call with Python

```python
import requests
import json

url = "http://localhost:8000/predict"
payload = {
    "monthly_fee": 50,
    "total_amount": 500,
    "trans_count": 10,
    "recency_days": 30,
    "recency_days_tx": 10,
    "tenure_days": 365,
    "logins_7d": 5,
    "logins_30d": 20,
    "logins_90d": 60,
    "tickets": 2,
    "avg_resolution": 24,
    "plan": "standard"
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

## Troubleshooting

### Issue: Models not loading
**Solution:** Ensure models are trained and saved in the `models/` directory:
```bash
ls -la models/  # macOS/Linux
dir models\    # Windows
```

Required files:
- `churn_model.pkl`
- `cltv_model.pkl`
- `scaler.joblib`
- `encoder.joblib`

### Issue: Frontend can't connect to API
**Solution:** Check CORS configuration in `deploy_api.py`. Ensure the API is running on port 8000:
```bash
curl http://localhost:8000/health
```

### Issue: Port already in use
**Solution:** Use a different port:
```bash
# Change API port
python src/deploy_api.py --port 8001

# Change frontend port
python -m http.server 8081
```

### Issue: Permission denied on Windows PowerShell
**Solution:** Run with explicit interpreter:
```bash
python -m uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
```

### Issue: Docker build fails
**Solution:** Clear Docker cache and rebuild:
```bash
docker system prune -a
docker-compose up --build
```

## Development Tips

### Enable Auto-Reload for API Development
```bash
python -m uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
```

### Debug Mode
Set environment variable:
```bash
# macOS/Linux
export DEBUG=True

# Windows
set DEBUG=True
```

### View API Logs
```bash
# Real-time logs
tail -f deploy_api.log

# Windows PowerShell
Get-Content -Path deploy_api.log -Wait
```

## Dependencies

Key Python packages:
- **fastapi** - Web framework for API
- **uvicorn** - ASGI server
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning models
- **joblib** - Model serialization
- **numpy** - Numerical computing

See `requirements.txt` for complete list.

## Performance Considerations

- **API Response Time:** ~50-100ms per prediction
- **Concurrent Requests:** Handles 100+ concurrent users
- **Memory Usage:** ~500MB (models + data)
- **Storage:** ~100MB (trained models)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support & Contact
For issues, questions, or suggestions, please open an issue in the repository.

*Last Updated: December 2025*
*Project Version: 1.0.0*
