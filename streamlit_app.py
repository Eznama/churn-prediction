# streamlit_app.py
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

MODELS_DIR = Path("models")
PIPE_PATH = MODELS_DIR / "logistic_pipeline.joblib"
TH_PATH = MODELS_DIR / "threshold.json"

@st.cache_resource
def load_artifacts():
    pipe = joblib.load(PIPE_PATH)
    meta = json.loads(TH_PATH.read_text(encoding="utf-8"))
    return pipe, float(meta.get("threshold", 0.5))

pipe, threshold = load_artifacts()

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption(f"Using logistic regression pipeline · decision threshold = **{threshold:.2f}**")

with st.form("form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
        MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])

    with col2:
        DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
        PaymentMethod = st.selectbox("PaymentMethod",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure = st.number_input("tenure (months)", min_value=0, max_value=1000, value=2, step=1)
        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=85.0, step=1.0)
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=190.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
    }
    X = pd.DataFrame([payload])
    prob = float(pipe.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    st.markdown("### Result")
    st.metric("Churn probability", f"{prob:.2%}")
    st.write("**Prediction:**", "⚠️ Likely to churn" if pred else "✅ Unlikely to churn")
