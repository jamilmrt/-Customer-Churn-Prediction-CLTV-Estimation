"""Training script
Trains churn (classifier) and CLTV (regressor) and saves models to ../models
Usage: python src/train.py --input intermediate/ --models models/
"""
import pandas as pd, joblib, argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

def run(intermediate_dir, models_dir):
    intermediate = Path(intermediate_dir)
    models = Path(models_dir); models.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(intermediate/"engineered_features.csv")
    labels = pd.read_csv(intermediate/"labels.csv")

    num_cols = ["monthly_fee","total_amount","trans_count","recency_days","recency_days_tx","logins_7d","logins_30d","logins_90d","tickets","avg_resolution","tenure_days"]
    X_num = features[num_cols].fillna(0)
    le = LabelEncoder()
    features["plan_enc"] = le.fit_transform(features["plan"])
    X_num["plan_enc"] = features["plan_enc"]

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    joblib.dump(scaler, models/"scaler.joblib")
    joblib.dump(le, models/"encoder.joblib")

    # train churn classifier
    y = features["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    print("Churn model ROC-AUC:", round(auc,4))
    joblib.dump(clf, models/"churn_model.pkl")

    # train CLTV regressor
    Xr = X_num.copy(); yr = labels["future_12m_revenue"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(Xr_train, yr_train)
    preds = reg.predict(Xr_test)
    rmse = mean_squared_error(yr_test, preds, squared=False)
    print("CLTV RMSE:", round(rmse,4))
    joblib.dump(reg, models/"cltv_model.pkl")
    print(f"Saved models to {models}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="intermediate", help="Intermediate dir")
    parser.add_argument("--models", default="models", help="Models dir")
    args = parser.parse_args()
    run(args.input, args.models)
