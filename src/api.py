from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

try:
    pipe = joblib.load(MODEL_PATH)
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"]
except Exception as e:
    raise RuntimeError(f"Failed to load model assets: {e}")

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

def _payload_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame([d])

app = FastAPI(title="Churn API", version="0.1")

@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}

@app.post("/predict")
def predict(customer: Customer):
    try:
        X = _payload_to_df(customer.model_dump())
        prob = float(pipe.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)
        return {"threshold": threshold, "prob_churn": prob, "churn_pred": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
