# Customer Churn Prediction and CLTV Estimation with Explainable AI

This repository contains the source code for a final year project focused on predicting customer churn and estimating Customer Lifetime Value (CLTV). The project implements a machine learning pipeline to process data, train predictive models, and serve the results via an API.

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started (Local Development)](#getting-started-local-development)
- [Usage (Running the Pipeline Locally)](#usage-running-the-pipeline-locally)
- [Docker & Deployment](#docker--deployment)
- [License](#license)

## Project Structure
The repository is organized to separate data, source code, models, and outputs for clarity and reproducibility.

- `inputs/`: Contains raw input data (CSV files).
- `intermediate/`: Stores intermediate data files, such as engineered features.
- `models/`: Stores trained model artifacts (e.g., `churn_model.pkl`, `encoder.joblib`).
- `outputs/`: Contains the final outputs of the pipeline, like scored customer lists.
- `src/`: Contains the core Python source code for the pipeline.
  - `feature_engineering.py`: Script for data preprocessing and feature creation.
  - `train.py`: Script for training the churn and CLTV models.
  - `score.py`: Script to score customers using the trained models.
  - `deploy_api.py`: A Flask API to serve predictions.
- `deployment/`: Contains alternative Docker configurations.
- `docker-compose.yml`: Main Docker Compose file for managing the application services.
- `requirements.txt`: A list of Python dependencies for the project.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8+
- Docker (optional, for containerized deployment)
- Git (optional, for cloning the repository)

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
- **Windows:**
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```

### 3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

## Usage (Running the Pipeline Locally)
The project is divided into several steps. Run them in the following order:

### 1. Run Feature Engineering
This script processes the raw data from the `inputs/` directory and saves the engineered features to the `intermediate/` directory.
```bash
python src/feature_engineering.py
```

### 2. Train the Models
This script uses the engineered features to train the churn and CLTV models, saving the artifacts to the `models/` directory.
```bash
python src/train.py
```

### 3. Score the Customers
This script applies the trained models to generate scores for customers.
```bash
python src/score.py
```

### 4. Run the API Server
To serve the model's predictions via a REST API, run the following command. The API will be accessible at `http://127.0.0.1:5000`.
```bash
python src/deploy_api.py
```

## Docker & Deployment
For consistent deployment and easy management, you can use Docker and Docker Compose. The top-level `docker-compose.yml` is configured for this purpose.

### Running with Docker Compose
To build and run the entire application stack (including the API service), use the following command from the project root directory:
```bash
docker-compose up --build
```
To run the services in the background, use the `-d` flag:
```bash
docker-compose up --build -d
```
To stop the services:
```bash
docker-compose down
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*This README has been updated for clarity and professionalism.*
