# analysis.py  (or paste into notebook cells)

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# If you want nicer plots, uncomment:
# import seaborn as sns
# sns.set(style="whitegrid")

print("✅ Libraries loaded")


# 2. Load the 4 input files
customers = pd.read_csv("inputs/customers.csv", parse_dates=["signup_date"])
transactions = pd.read_csv("inputs/transactions.csv", parse_dates=["transaction_date"])
events = pd.read_csv("inputs/events.csv", parse_dates=["event_date"])
support = pd.read_csv("inputs/support.csv", parse_dates=["ticket_date"])

print("✅ Data loaded\n")

print("Customers (top 5):")
print(customers.head(), "\n")

print("Transactions (top 5):")
print(transactions.head(), "\n")

print("Events (top 5):")
print(events.head(), "\n")

print("Support (top 5):")
print(support.head(), "\n")


# 3. Shapes of each dataset
print("📏 SHAPES")
print("Customers:", customers.shape)
print("Transactions:", transactions.shape)
print("Events:", events.shape)
print("Support:", support.shape, "\n")


# 4. Missing values
print("❓ MISSING VALUES\n")

def missing_report(df, name):
    print(f"--- {name} ---")
    print(df.isnull().sum())
    print()

missing_report(customers, "customers")
missing_report(transactions, "transactions")
missing_report(events, "events")
missing_report(support, "support")


# 5. Basic statistics on numerical columns (customers)
print("📊 Customers describe():")
print(customers.describe(include="all"))


# 6. Simple EDA plots (run only if in notebook / interactive)
def quick_plots():
    # Plan distribution if 'plan' column exists
    if "plan" in customers.columns:
        customers["plan"].value_counts().plot(kind="bar")
        plt.title("Plan Distribution")
        plt.xlabel("Plan")
        plt.ylabel("Count")
        plt.show()

    # Churn distribution if 'churn' column exists
    if "churn" in customers.columns:
        customers["churn"].value_counts().plot(kind="bar")
        plt.title("Churn Distribution")
        plt.xlabel("Churn (0 = no, 1 = yes)")
        plt.ylabel("Count")
        plt.show()

    # Transaction amount distribution
    if "amount" in transactions.columns:
        transactions["amount"].hist(bins=40)
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.show()

# Uncomment this if running in Jupyter or want to see plots:
# quick_plots()


# 7. Simple feature engineering preview:
#    - Total amount per customer
#    - Transaction count per customer
#    - Last transaction date
print("\n🧮 Feature engineering preview...")

trans_agg = (
    transactions
    .groupby("customer_id")
    .agg(
        total_amount=("amount", "sum"),
        trans_count=("amount", "count"),
        last_trans_date=("transaction_date", "max"),
    )
    .reset_index()
)

print("Transaction aggregates (top 5):")
print(trans_agg.head(), "\n")

# Merge with customers as an example
customers_fe = customers.merge(trans_agg, on="customer_id", how="left")

print("Customers + basic features (top 5):")
print(customers_fe.head(), "\n")


# 8. Simple model example: churn prediction using few features
#    (for proper training, use train.py later)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("🤖 Simple churn model demo...")

df = customers_fe.copy()

# Drop rows where churn is missing (if any)
df = df[~df["churn"].isna()]

# Encode plan if exists
if "plan" in df.columns:
    le_plan = LabelEncoder()
    df["plan_enc"] = le_plan.fit_transform(df["plan"])
else:
    df["plan_enc"] = 0  # fallback

# Fill NaNs from feature engineering
for col in ["total_amount", "trans_count"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Select a small set of features
feature_cols = []
for col in ["monthly_fee", "recency_days", "total_amount", "trans_count", "plan_enc"]:
    if col in df.columns:
        feature_cols.append(col)

X = df[feature_cols]
y = df["churn"].astype(int)

print("Using features:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape, "\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Classification report:")
print(classification_report(y_test, y_pred))

try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", round(auc, 4))
except Exception:
    pass

print("\n✅ Notebook-style analysis done.")