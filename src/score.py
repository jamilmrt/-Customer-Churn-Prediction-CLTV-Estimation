"""Scoring script
Loads engineered features and models, produces scored_customers.csv and retention_list.csv in outputs/
Usage: python src/score.py --intermediate intermediate/ --models models/ --output outputs/
"""
import pandas as pd, joblib, argparse, numpy as np
from pathlib import Path

def run(intermediate_dir, models_dir, output_dir):
    inter = Path(intermediate_dir)
    models = Path(models_dir)
    outp = Path(output_dir); outp.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(inter/"engineered_features.csv")
    num_cols = ["monthly_fee","total_amount","trans_count","recency_days","recency_days_tx","logins_7d","logins_30d","logins_90d","tickets","avg_resolution","tenure_days"]

    X_num = features[num_cols].fillna(0)
    le = joblib.load(models/"encoder.joblib")
    X_num["plan_enc"] = le.transform(features["plan"])
    scaler = joblib.load(models/"scaler.joblib")
    X_scaled = scaler.transform(X_num)

    clf = joblib.load(models/"churn_model.pkl")
    reg = joblib.load(models/"cltv_model.pkl")

    churn_probs = clf.predict_proba(X_scaled)[:,1]
    cltv_preds = reg.predict(X_num)

    scored = pd.DataFrame({
        "customer_id": features["customer_id"],
        "churn_prob": churn_probs,
        "cltv_pred": cltv_preds
    })

    # approximate top feature per customer using feature importances and deviations
    feat_names = num_cols + ["plan_enc"]
    imp = dict(zip(feat_names, clf.feature_importances_))
    medians = X_num.median()
    corrs = {}
    for f in feat_names:
        try:
            corrs[f] = np.sign(np.corrcoef(X_num[f], features['churn'])[0,1])
        except:
            corrs[f] = 1.0
    top_feats = []
    for _, row in X_num.iterrows():
        scores = {f: imp.get(f,0.0) * (row[f] - medians[f]) * corrs.get(f,1.0) for f in feat_names}
        top = max(scores.items(), key=lambda x: abs(x[1]))[0]
        top_feats.append(top)
    scored["top_shap_feature"] = top_feats

    scored.to_csv(outp/"scored_customers.csv", index=False)
    scored["priority_score"] = scored["churn_prob"] * scored["cltv_pred"]
    retention = scored.sort_values("priority_score", ascending=False).head(200)
    retention.to_csv(outp/"retention_list.csv", index=False)
    print(f"Saved scored_customers.csv and retention_list.csv to {outp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", default="intermediate", help="Intermediate dir")
    parser.add_argument("--models", default="models", help="models dir")
    parser.add_argument("--output", default="outputs", help="output dir")
    args = parser.parse_args()
    run(args.intermediate, args.models, args.output)
