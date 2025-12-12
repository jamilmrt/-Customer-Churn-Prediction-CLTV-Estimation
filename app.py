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
        'churn_prob': churn_prob,
        'cltv_pred': cltv_pred,
        # 'churn_explanation': churn_explanation,
        # 'cltv_explanation': cltv_explanation
    })

@app.route('/explain', methods=['POST'])
def explain():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "The GEMINI_API_KEY environment variable is not set."}), 500
    
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        return jsonify({"error": f"Failed to configure Gemini API: {e}"}), 500

    data = request.get_json()
    inputs = data.get('inputs', {})
    predictions = data.get('predictions', {})
    churn_prob = predictions.get('churn_prob', 0)
    
    prompt = f"""
    You are an expert business analyst explaining a customer churn prediction to a manager.
    The output must be in three parts, separated by '---'.

    **Customer Data:**
    - Tenure: {inputs.get('tenure_days')} days
    - Monthly Fee: ${inputs.get('monthly_fee')}
    - Logins (30 days): {inputs.get('logins_30d')}
    - Support Tickets: {inputs.get('tickets')}
    - Plan Type: {inputs.get('plan')}
    - Recency (days since last transaction): {inputs.get('recency_days')}

    **Prediction Results:**
    - Churn Probability: {churn_prob:.2%}
    - Predicted Customer Lifetime Value (CLTV): ${predictions.get('cltv_pred', 0):.2f}
    ---
    **Part 1: Summary**
    Provide a one-paragraph summary explaining what this churn probability means for this specific customer and the potential business impact.
    ---
    **Part 2: Actionable Recommendations**
    Provide a list of 3-4 specific, actionable recommendations for the business to retain this customer. Use bullet points.
    ---
    **Part 3: Key Drivers**
    Identify the top 3 key drivers for this prediction from the provided customer data. Format this strictly as a JSON object string like this: {{\"labels\": [\"Driver 1\", \"Driver 2\", \"Driver 3\"], \"data\": [85, 60, 45]}}. 'data' is an impact score from 0-100. Do not write anything before or after this single-line JSON string.
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Handle potential safety blocks
        if not response.parts:
            return jsonify({"error": "The response from the AI was blocked for safety reasons."}), 500

        # The prompt asks for '---' as a separator.
        # It's safer to expect the model might not follow instructions perfectly.
        # We look for the JSON part specifically.
        import re
        json_match = re.search(r'\{.*\}', response.text)
        if not json_match:
            raise ValueError("Could not find the JSON part for the chart in the AI response.")
        
        visual_data_str = json_match.group(0)
        visual_data = json.loads(visual_data_str)
        
        # The text part is everything else.
        text_part = response.text.replace(visual_data_str, "").replace('---', '<hr>').strip()
        
        # Basic formatting for HTML display
        text_part = text_part.replace('**', '<strong>').replace('</strong>', '</strong><br>')
        text_part = text_part.replace('* ', '<br>&bull; ')
        
        return jsonify({
            "text": text_part,
            "visual": visual_data
        })

    except Exception as e:
        print(f"Gemini API or parsing error: {e}")
        return jsonify({"error": "Failed to process the explanation from the AI. Please check the server logs."}), 500

if __name__ == '__main__':
    app.run(debug=True)