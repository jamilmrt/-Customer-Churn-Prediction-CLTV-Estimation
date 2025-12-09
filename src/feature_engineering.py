"""Feature engineering script
Reads input CSVs from ../inputs and writes engineered_features.csv and labels.csv to ../intermediate
Usage: python src/feature_engineering.py --inputs inputs/ --output intermediate/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def run(inputs_dir, output_dir, reference_date="2024-01-01"):
    inputs = Path(inputs_dir)
    outputs = Path(output_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(inputs/"customers.csv", parse_dates=["signup_date"])
    transactions = pd.read_csv(inputs/"transactions.csv", parse_dates=["transaction_date"])
    events = pd.read_csv(inputs/"events.csv", parse_dates=["event_date"])
    support = pd.read_csv(inputs/"support.csv", parse_dates=["ticket_date"]) if (inputs/"support.csv").exists() else pd.DataFrame()

    ref = pd.to_datetime(reference_date)

    # Transaction aggregates
    trans_agg = transactions.groupby("customer_id").agg(
        total_amount=("amount","sum"),
        trans_count=("amount","count"),
        last_trans_date=("transaction_date","max"),
        first_trans_date=("transaction_date","min")
    ).reset_index()
    trans_agg["recency_days_tx"] = (ref - pd.to_datetime(trans_agg["last_trans_date"])).dt.days.fillna(999)

    # Events aggregates (logins in windows)
    events["days_from_ref"] = (ref - events["event_date"]).dt.days
    logins_7 = events[events["days_from_ref"]<=7].groupby("customer_id").size().rename("logins_7d")
    logins_30 = events[events["days_from_ref"]<=30].groupby("customer_id").size().rename("logins_30d")
    logins_90 = events[events["days_from_ref"]<=90].groupby("customer_id").size().rename("logins_90d")
    logins = pd.concat([logins_7, logins_30, logins_90], axis=1).fillna(0).reset_index()

    # Support aggregates
    if not support.empty:
        support_agg = support.groupby("customer_id").agg(tickets=("issue_type","count"), avg_resolution=("resolution_days","mean")).reset_index()
    else:
        support_agg = pd.DataFrame(columns=["customer_id","tickets","avg_resolution"])

    # Merge
    features = customers.merge(trans_agg, on="customer_id", how="left").merge(logins, on="customer_id", how="left").merge(support_agg, on="customer_id", how="left")
    # Fill missing
    features["total_amount"] = features["total_amount"].fillna(0)
    features["trans_count"] = features["trans_count"].fillna(0).astype(int)
    features["logins_7d"] = features["logins_7d"].fillna(0).astype(int)
    features["logins_30d"] = features["logins_30d"].fillna(0).astype(int)
    features["logins_90d"] = features["logins_90d"].fillna(0).astype(int)
    features["tickets"] = features["tickets"].fillna(0).astype(int)
    features["avg_resolution"] = features["avg_resolution"].fillna(0)
    features["signup_date"] = pd.to_datetime(features["signup_date"])
    features["tenure_days"] = (ref - features["signup_date"]).dt.days

    # select columns to save
    out = features[["customer_id","plan","monthly_fee","total_amount","trans_count","recency_days","recency_days_tx",
                    "logins_7d","logins_30d","logins_90d","tickets","avg_resolution","tenure_days","churn"]].copy()

    out.to_csv(outputs/"engineered_features.csv", index=False)

    # Labels (simulate CLTV if not provided)
    labels = out[["customer_id","churn"]].copy()
    rng = np.random.default_rng(12345)
    future_months = rng.poisson(6, size=len(labels)) + 1
    labels["future_12m_revenue"] = out["monthly_fee"] * future_months
    labels.to_csv(outputs/"labels.csv", index=False)
    print(f"Saved engineered features to {outputs/'engineered_features.csv'} and labels to {outputs/'labels.csv'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="inputs", help="Inputs directory (contains customers.csv etc)")
    parser.add_argument("--output", default="intermediate", help="Output directory for engineered features")
    args = parser.parse_args()
    run(args.inputs, args.output)