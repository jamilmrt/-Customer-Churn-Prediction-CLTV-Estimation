"""
FastAPI app for Churn + CLTV model serving.

Expect model artifacts in MODEL_DIR (default: ./models):
 - churn_model.pkl        (classifier with predict_proba)
 - cltv_model.pkl         (regressor with predict)
 - scaler.joblib          (sklearn scaler used in training)
 - encoder.joblib         (LabelEncoder or similar for 'plan')

Endpoints:
 - GET  /health
 - POST /predict  (JSON body containing required features)

Run:
    export MODEL_DIR=./models
    uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import sys
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deploy_api.log')
    ]
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
logger.info(f"MODEL_DIR set to: {MODEL_DIR}")

# --- Request schema ---
class PredictRequest(BaseModel):
    monthly_fee: float = Field(..., json_schema_extra={"example": 20.0})
    total_amount: float = Field(..., json_schema_extra={"example": 240.0})
    trans_count: int = Field(..., json_schema_extra={"example": 12})
    recency_days: float = Field(..., json_schema_extra={"example": 10.0})
    recency_days_tx: Optional[float] = Field(None, json_schema_extra={"example": 10.0})
    tenure_days: Optional[float] = Field(None, json_schema_extra={"example": 400.0})
    logins_7d: Optional[int] = Field(0, json_schema_extra={"example": 3})
    logins_30d: Optional[int] = Field(0, json_schema_extra={"example": 12})
    logins_90d: Optional[int] = Field(0, json_schema_extra={"example": 35})
    tickets: Optional[int] = Field(0, json_schema_extra={"example": 0})
    avg_resolution: Optional[float] = Field(0.0, json_schema_extra={"example": 1.2})
    plan: Optional[str] = Field(None, json_schema_extra={"example": "standard"})

# --- Response schema ---
class PredictResponse(BaseModel):
    churn_prob: Optional[float] = None
    cltv_pred: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None

# Globals for loaded artifacts
clf = None
reg = None
scaler = None
encoder = None
feature_order = [
    "monthly_fee","total_amount","trans_count","recency_days",
    "recency_days_tx","logins_7d","logins_30d","logins_90d",
    "tickets","avg_resolution","tenure_days","plan_enc"
]

def _safe_load(p: Path):
    """Safely load a joblib file with error handling."""
    logger.info(f"Attempting to load: {p}")
    if not p.exists():
        logger.warning(f"File does not exist: {p}")
        return None
    try:
        obj = joblib.load(p)
        logger.info(f"Successfully loaded: {p}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {p}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_models():
    """Load all required models from MODEL_DIR."""
    global clf, reg, scaler, encoder
    
    logger.info("=" * 60)
    logger.info("Starting model loading...")
    logger.info("=" * 60)
    
    base = Path(MODEL_DIR)
    logger.info(f"Model base directory: {base}")
    logger.info(f"Base directory exists: {base.exists()}")
    
    if base.exists():
        logger.info(f"Contents of {base}: {list(base.glob('*'))}")
    
    clf = _safe_load(base / "churn_model.pkl")
    reg = _safe_load(base / "cltv_model.pkl")
    scaler = _safe_load(base / "scaler.joblib")
    encoder = _safe_load(base / "encoder.joblib")
    
    logger.info("=" * 60)
    logger.info(f"Models loaded - Churn: {clf is not None}, CLTV: {reg is not None}")
    logger.info(f"Scaler: {scaler is not None}, Encoder: {encoder is not None}")
    logger.info("=" * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Application startup - loading models...")
    load_models()
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="Churn + CLTV Prediction API",
    version="1.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Health check endpoint."""
    logger.info("Health check called")
    return {
        "status": "ok",
        "models_loaded": {
            "churn_model": clf is not None,
            "cltv_model": reg is not None,
            "scaler": scaler is not None,
            "encoder": encoder is not None
        },
        "model_dir": MODEL_DIR
    }

def build_feature_vector(req: PredictRequest) -> pd.DataFrame:
    """Build a single-row DataFrame with the same feature order used in training."""
    logger.debug(f"Building feature vector from request: {req}")
    
    d = req.dict()
    row = {
        "monthly_fee": float(d.get("monthly_fee", 0.0)),
        "total_amount": float(d.get("total_amount", 0.0)),
        "trans_count": int(d.get("trans_count", 0)),
        "recency_days": float(d.get("recency_days", 0.0)),
        "recency_days_tx": float(d.get("recency_days_tx") or 0.0),
        "logins_7d": int(d.get("logins_7d") or 0),
        "logins_30d": int(d.get("logins_30d") or 0),
        "logins_90d": int(d.get("logins_90d") or 0),
        "tickets": int(d.get("tickets") or 0),
        "avg_resolution": float(d.get("avg_resolution") or 0.0),
        "tenure_days": float(d.get("tenure_days") or 0.0),
    }
    
    plan = d.get("plan", None)
    if encoder is not None and plan is not None:
        try:
            plan_enc = int(encoder.transform([plan])[0])
            logger.debug(f"Encoded plan '{plan}' to {plan_enc}")
        except Exception as e:
            logger.warning(f"Failed to encode plan '{plan}': {e}")
            plan_enc = 0
    else:
        plan_enc = 0
    
    row["plan_enc"] = plan_enc
    df = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_order}])
    logger.debug(f"Feature vector shape: {df.shape}, columns: {list(df.columns)}")
    return df

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Prediction endpoint."""
    logger.info(f"Prediction request received")
    
    try:
        if clf is None and reg is None:
            logger.error("No models loaded on server")
            raise HTTPException(status_code=503, detail="No models loaded on server")

        X = build_feature_vector(req)
        
        X_scaled = X.values
        try:
            if scaler is not None:
                X_scaled = scaler.transform(X)
                logger.debug("Applied scaler transformation")
        except Exception as e:
            logger.warning(f"Scaler transformation failed, using raw values: {e}")
            X_scaled = X.values

        churn_prob = None
        cltv_pred = None
        explanation = {}

        if clf is not None:
            try:
                if hasattr(clf, "predict_proba"):
                    churn_prob = float(clf.predict_proba(X_scaled)[0, 1])
                else:
                    churn_prob = float(clf.predict(X_scaled)[0])
                explanation["churn_model"] = {"type": type(clf).__name__}
                logger.info(f"Churn prediction: {churn_prob}")
            except Exception as e:
                logger.error(f"Churn prediction failed: {e}")
                logger.error(traceback.format_exc())
                explanation["churn_error"] = str(e)

        if reg is not None:
            try:
                cltv_pred = float(reg.predict(X_scaled)[0])
                explanation["cltv_model"] = {"type": type(reg).__name__}
                logger.info(f"CLTV prediction: {cltv_pred}")
            except Exception as e:
                logger.error(f"CLTV prediction failed: {e}")
                logger.error(traceback.format_exc())
                explanation["cltv_error"] = str(e)

        return PredictResponse(churn_prob=churn_prob, cltv_pred=cltv_pred, explanation=explanation)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error: {str(exc)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")

if __name__ == "__main__":
    logger.info("Running deploy_api directly")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

