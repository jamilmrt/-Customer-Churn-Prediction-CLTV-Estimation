import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
import shap
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models and preprocessors
churn_model = joblib.load('models/churn_model.pkl')
cltv_model = joblib.load('models/cltv_model.pkl')
scaler = joblib.load('models/scaler.joblib')
encoder = joblib.load('models/encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])

    # Preprocess features
    num_cols = ["monthly_fee", "total_amount", "trans_count", "recency_days", "recency_days_tx", "logins_7d", "logins_30d", "logins_90d", "tickets", "avg_resolution", "tenure_days"]
    X_num = features[num_cols].fillna(0)
    
    # Handle 'plan' feature
    try:
        features["plan_enc"] = encoder.transform(features["plan"])
    except ValueError:
        # If the plan is not in the encoder, use a default value (e.g., -1 or the most frequent class)
        # For simplicity, we'll use -1, but a more robust solution might be needed
        features["plan_enc"] = -1

    X_num["plan_enc"] = features["plan_enc"]


    X_scaled = scaler.transform(X_num)

    # Churn prediction
    churn_prob = churn_model.predict_proba(X_scaled)[:, 1][0]

    # CLTV prediction
    cltv_pred = cltv_model.predict(X_num)[0]

    # Churn explainability
    churn_explainer = shap.TreeExplainer(churn_model)
    churn_shap_values = churn_explainer.shap_values(X_scaled)
    
    # It looks like the shap values array is multi-dimensional, so we need to get the correct slice
    if isinstance(churn_shap_values, list) and len(churn_shap_values) == 2:
        churn_shap_values_slice = churn_shap_values[1]
    else:
        churn_shap_values_slice = churn_shap_values
        
    plt.figure(figsize=(10, 5))
    shap.force_plot(churn_explainer.expected_value[1], churn_shap_values_slice[0, :], X_num.iloc[0,:], matplotlib=True, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    churn_explanation = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # CLTV explainability
    cltv_explainer = shap.TreeExplainer(cltv_model)
    cltv_shap_values = cltv_explainer.shap_values(X_num)
    plt.figure(figsize=(10, 5))
    shap.force_plot(cltv_explainer.expected_value, cltv_shap_values[0, :], X_num.iloc[0,:], matplotlib=True, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    cltv_explanation = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()


    return jsonify({
        'churn_prediction': churn_prob,
        'cltv_prediction': cltv_pred,
        'churn_explanation': churn_explanation,
        'cltv_explanation': cltv_explanation
    })

if __name__ == '__main__':
    app.run(debug=True)