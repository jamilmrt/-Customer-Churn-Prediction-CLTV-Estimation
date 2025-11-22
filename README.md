Full Project Code Structure

How to run locally (quick):
1. Place your input CSVs into the 'inputs/' directory:
   - customers.csv, transactions.csv, events.csv, support.csv

2. Create virtual env and install dependencies:
   python -m venv venv
   source venv/bin/activate   # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt

3. Run feature engineering:
   python src/feature_engineering.py --inputs inputs --output intermediate

4. Train models:
   python src/train.py --input intermediate --models models

5. Score customers: 
   python src/score.py --intermediate intermediate --models models --output outputs

6. Run API (optional):
   uvicorn src.deploy_api:app --reload --host 0.0.0.0 --port 8000
