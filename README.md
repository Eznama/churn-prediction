# Churn Prediction (Telco)

![CI](https://github.com/Eznama/churn-prediction/actions/workflows/ci.yml/badge.svg?branch=main)

End-to-end, reproducible churn-prediction project on the classic Telco dataset.
Includes EDA, a scikit-learn preprocessing + model pipeline, evaluation, a tiny CLI for scoring one customer from JSON, and GitHub Actions smoke tests.

---

## Project structure

```
churn-prediction/
├─ data/
│  └─ raw/
│     └─ telco_churn.csv                # input data (notebook expects this path)
├─ notebooks/
│  └─ 01_eda_and_model.ipynb            # EDA → preprocessing → training → evaluation → export
├─ src/
│  └─ predict.py                         # CLI to score one customer JSON
├─ models/
│  ├─ logistic_pipeline.joblib           # fitted ColumnTransformer + LogisticRegression
│  └─ threshold.json                     # operating threshold metadata (e.g., 0.54)
├─ samples/
│  ├─ customer_risky.json                # example likely churner
│  └─ customer_safe.json                 # example likely non-churner
├─ reports/
│  ├─ figures/                           # ROC, PR, confusion matrices, feature plots
│  └─ tables/                            # coefficients.csv, permutation_importance*.csv
├─ .github/workflows/ci.yml              # CI: smoke tests for CLI
├─ requirements.txt
└─ README.md
```

---

## Quickstart (Local Demo)

### Start the API (pulls the public Docker image)
```bash
docker run -d --name churn-api -p 8010:8000 eznama/churn-api:v0.3

> **Windows PowerShell**

```powershell
# 1) Create & activate a virtual env
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt
```

> **macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the notebook (optional, to reproduce)

Open `notebooks/01_eda_and_model.ipynb` in JupyterLab. It expects the CSV at `data/raw/telco_churn.csv`.
The notebook:

* loads & cleans data (normalizes labels, coerces numeric fields);
* builds a preprocessing pipeline
  **Numeric:** `SimpleImputer(median)` → `StandardScaler`
  **Categorical:** `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
* trains **LogisticRegression** (class\_weight="balanced") and a **RandomForest** baseline;
* evaluates (ROC/PR, confusion matrices at default & tuned threshold);
* chooses an operating threshold for the churn class (max F1);
* exports the fitted pipeline and threshold to `models/`.

---

## CLI: score one customer

The CLI reads a single customer JSON and writes a JSON with churn probability and decision.

```powershell
# High-risk example
python src/predict.py --input samples/customer_risky.json --out outputs/risky.json
Get-Content outputs\risky.json

# Low-risk example
python src/predict.py --input samples/customer_safe.json --out outputs/safe.json
Get-Content outputs\safe.json
```

**Output shape**

```json
{
  "threshold": 0.54,
  "prob_churn": 0.9076,
  "churn_pred": 1
}
```

**Required JSON fields**

```
SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
gender, Partner, Dependents, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod
```

(See `samples/*.json` for valid values.)

---

## Results (baseline)

* **Churn rate:** \~**26.5%** (No: 5174, Yes: 1869)

**Logistic Regression**

* Test ROC AUC ≈ **0.84**
* Best F1 (churn class) ≈ **0.62** at **threshold 0.54**
* Confusion matrix @0.54 (test): TN ≈ 775, FP ≈ 260, FN ≈ 88, TP ≈ 286
  Precision(1) ≈ 0.52 · Recall(1) ≈ 0.77
* Top churn ↑ drivers (coef): **Fiber optic**, **Month-to-month**, **TotalCharges**, StreamingMovies=Yes, StreamingTV=Yes
  Churn ↓ drivers: **tenure**, **Two-year contract**, MonthlyCharges (model-specific effect)

**Random Forest**

* CV ROC AUC ≈ **0.837 ± 0.009** · Test ROC AUC ≈ **0.834**
* Best F1(1) ≈ **0.625** @ threshold ≈ **0.38** (Precision~~0.53, Recall~~0.77)

**Permutation importance (AUC drop):** **tenure**, **InternetService**, **Contract**, **MonthlyCharges**, **TotalCharges**

Plots in `reports/figures/`, tables in `reports/tables/`.

---

## CI

* GitHub Actions workflow (`.github/workflows/ci.yml`) smoke-tests the CLI:

  * sets up Python,
  * installs `requirements.txt`,
  * runs `src/predict.py` on the two sample JSONs,
  * verifies outputs exist and prints them.
* The badge at the top shows current status for `main`.

---

## Notes & limitations

* **Threshold** chosen for **F1** on churn; in production, pick a threshold that reflects business costs (false positives vs. false negatives) and consider probability calibration.
* **Correlated features:** tenure ↔ contract; interpret coefficients with care. Prefer permutation importance / SHAP for explanations.
* **Imbalance:** \~26.5% churn handled with class weights. Alternatives: resampling, focal loss, cost-sensitive metrics.
* **Validation:** current split + CV for RF; consider nested CV / hold-out for stronger estimates.

---

## Reproducibility

Re-training the notebook overwrites:

* `models/logistic_pipeline.joblib`
* `models/threshold.json`

The CLI always loads these; artifacts are committed so CI can run.

---

## Local Demo (API + Dashboard)

### Prereqs
- Docker Desktop
- Python 3.10+ (3.12 recommended)
- PowerShell (Windows) or a shell (macOS/Linux)

### 1) Start the API (Docker)
```powershell
docker rm -f churn-api 2>$null
docker run -d --name churn-api -p 8010:8000 eznama/churn-api:v0.3

Health check:

Invoke-RestMethod http://127.0.0.1:8010/health

### 2) Start the Streamlit dashboard

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run .\streamlit_app.py --server.port 8501

Open: http://localhost:8501

Click Check API health, then Send risky sample / Send safe sample.
You can also paste/edit JSON and click Predict with JSON above.

Troubleshooting

If port 8010 is busy: docker rm -f churn-api then re-run the docker run command.

If streamlit isn’t found: always use python -m streamlit ....

If samples error with encoding, re-pull the repo or ensure the two JSON files are saved as UTF-8 (no BOM).


## 3) One-click scripts
- Start everything: `./start.ps1`
- Stop API container: `./stop.ps1`

---

## License

MIT
