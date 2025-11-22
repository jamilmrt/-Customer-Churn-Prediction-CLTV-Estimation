# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib, pandas as pd, os
# from pathlib import Path

# MODEL_DIR = os.getenv("MODEL_DIR", "models")

# app = FastAPI(title="Churn+CLTV API")

# class CustomerFeatures(BaseModel):
#     monthly_fee: float
#     total_amount: float
#     trans_count: int
#     recency_days: float
#     recency_days_tx: float = 0.0
#     age_days: float = 0.0
#     logins_7d: int = 0
#     logins_30d: int = 0
#     logins_90d: int = 0

# @app.on_event("startup")
# def load_models():
#     global clf, reg, scaler, encoder
#     base = Path(MODEL_DIR)
#     clf = joblib.load(base/"churn_model.pkl")
#     reg = joblib.load(base/"cltv_model.pkl")
#     scaler = joblib.load(base/"scaler.joblib")
#     encoder = joblib.load(base/"encoder.joblib")

# @app.get("/health")
# def health():
#     return {"status":"ok", "models_loaded": {"churn": clf is not None, "cltv": reg is not None} }

# @app.post("/predict")
# def predict(features: CustomerFeatures):
#     X = pd.DataFrame([features.dict()])
#     # build feature vector order consistent with training
#     num_cols = ["monthly_fee","total_amount","trans_count","recency_days","recency_days_tx","logins_7d","logins_30d","logins_90d","tickets","avg_resolution","tenure_days"]
#     # For simplicity, map available fields and fill missing defaults
#     X_num = pd.DataFrame({c: X[c] if c in X.columns else 0 for c in num_cols}).fillna(0)
#     # encoder for plan not provided here; expect pre-computed inputs for API use
#     X_scaled = scaler.transform(X_num)
#     churn_prob = float(clf.predict_proba(X_scaled)[0,1])
#     cltv_pred = float(reg.predict(X_num)[0])
#     return {"churn_prob": churn_prob, "cltv_pred": cltv_pred}
# src/deploy_api.py
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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os
import traceback

MODEL_DIR = os.getenv("MODEL_DIR", "models")

app = FastAPI(title="Churn + CLTV Prediction API", version="1.0")

# --- Request schema ---
class PredictRequest(BaseModel):
    # Core numeric features (must match training features & order)
    monthly_fee: float = Field(..., example=20.0)
    total_amount: float = Field(..., example=240.0)
    trans_count: int = Field(..., example=12)
    recency_days: float = Field(..., example=10.0)
    recency_days_tx: Optional[float] = Field(None, example=10.0)
    tenure_days: Optional[float] = Field(None, example=400.0)
    logins_7d: Optional[int] = Field(0, example=3)
    logins_30d: Optional[int] = Field(0, example=12)
    logins_90d: Optional[int] = Field(0, example=35)
    tickets: Optional[int] = Field(0, example=0)
    avg_resolution: Optional[float] = Field(0.0, example=1.2)
    # Optional categorical field (if used)
    plan: Optional[str] = Field(None, example="standard")

# --- Response schema ---
class PredictResponse(BaseModel):
    churn_prob: Optional[float]
    cltv_pred: Optional[float]
    explanation: Optional[Dict[str, Any]] = None

# Globals for loaded artifacts
clf = None       # classification model
reg = None       # regression model
scaler = None    # scaler object
encoder = None   # encoder (LabelEncoder for plan) or None
feature_order = [
    "monthly_fee","total_amount","trans_count","recency_days",
    "recency_days_tx","logins_7d","logins_30d","logins_90d",
    "tickets","avg_resolution","tenure_days","plan_enc"
]

def _safe_load(p: Path):
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        # try pickle load or raise so startup doesn't completely fail
        raise

@app.on_event("startup")
def load_models():
    global clf, reg, scaler, encoder
    base = Path(MODEL_DIR)
    try:
        clf = _safe_load(base / "churn_model.pkl")
    except Exception as e:
        clf = None
        app.logger = getattr(app, "logger", None)
    try:
        reg = _safe_load(base / "cltv_model.pkl")
    except Exception:
        reg = None
    try:
        scaler = _safe_load(base / "scaler.joblib")
    except Exception:
        scaler = None
    try:
        encoder = _safe_load(base / "encoder.joblib")
    except Exception:
        encoder = None

@app.get("/health")
def health():
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
    """
    Build a single-row DataFrame with the same feature order used in training.
    If 'plan' is present and encoder is available, add plan_enc; else use default 0.
    """
    # Map request to dict
    d = req.dict()
    # Ensure keys exist and numeric defaults
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
    # handle plan -> plan_enc if encoder exists
    plan = d.get("plan", None)
    if encoder is not None and plan is not None:
        try:
            plan_enc = int(encoder.transform([plan])[0])
        except Exception:
            # if unseen label, try adding fallback: encoder may not support unseen -> set to -1 or 0
            # safer to set to median or 0
            plan_enc = 0
    else:
        plan_enc = 0
    row["plan_enc"] = plan_enc

    # Create DataFrame with feature_order; missing features get 0
    df = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_order}])
    return df

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if clf is None and reg is None:
            raise HTTPException(status_code=503, detail="No models loaded on server")

        X = build_feature_vector(req)

        # If scaler exists, apply to numeric columns (exclude plan_enc if scaler trained without it)
        X_scaled = None
        try:
            if scaler is not None:
                # scaler expects the same column order used during training; here we assume scaler was fit on X[feature_order]
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
        except Exception as e:
            # if scaler transformation fails, fallback to raw values but warn
            X_scaled = X.values

        churn_prob = None
        cltv_pred = None
        explanation = {}

        if clf is not None:
            # many sklearn tree models accept raw arrays; ensure correct shape
            # if scaler used, pass scaled; otherwise pass X.values
            try:
                # prefer predict_proba when available
                if hasattr(clf, "predict_proba"):
                    churn_prob = float(clf.predict_proba(X_scaled)[0, 1])
                else:
                    churn_prob = float(clf.predict(X_scaled)[0])
                explanation["churn_model"] = {"type": type(clf).__name__}
            except Exception as e:
                explanation["churn_error"] = str(e)
                churn_prob = None

        if reg is not None:
            try:
                cltv_pred = float(reg.predict(X.values)[0])
                explanation["cltv_model"] = {"type": type(reg).__name__}
            except Exception as e:
                explanation["cltv_error"] = str(e)
                cltv_pred = None

        return PredictResponse(churn_prob=churn_prob, cltv_pred=cltv_pred, explanation=explanation)

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")
