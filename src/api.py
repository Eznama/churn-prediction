from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- locate model artifacts ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

# ---------- request schema ----------
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def payload_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame([d])

app = FastAPI(title="Churn Predictor API")

@app.on_event("startup")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    app.state.pipe = joblib.load(MODEL_PATH)

    if THRESHOLD_PATH.exists():
        meta = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        app.state.threshold = float(meta.get("threshold", 0.5))
    else:
        app.state.threshold = 0.5

@app.get("/health")
def health():
    return {"status": "ok", "threshold": app.state.threshold}

@app.post("/predict")
def predict(customer: Customer):
    try:
        pipe = app.state.pipe
        threshold = app.state.threshold
        X = payload_to_df(customer.model_dump())
        prob = float(pipe.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)
        return {"threshold": threshold, "prob_churn": prob, "churn_pred": pred}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
