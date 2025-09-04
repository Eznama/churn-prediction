# Churn Prediction (Telco)

Minimal churn model built on the Telco Customer Churn dataset. Includes:
- Reproducible notebook (EDA, baseline models, evaluation)
- Saved pipeline + decision threshold
- Tiny CLI for single-customer predictions

## Quickstart
```bash
# 1) Clone & enter
git clone https://github.com/Eznama/churn-prediction.git
cd churn-prediction

# 2) Create venv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 3) Install deps
pip install -r requirements.txt

# 4) Try the CLI
python src/predict.py --input samples/customer_safe.json  --out outputs/safe.json
python src/predict.py --input samples/customer_risky.json --out outputs/risky.json
type outputs\safe.json
type outputs\risky.json
