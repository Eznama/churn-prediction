import os
import json
from pathlib import Path

import requests
import streamlit as st


# ---------- config ----------
DEFAULT_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8010")

BASE_DIR = Path(__file__).parent
RISKY = BASE_DIR / "samples" / "customer_risky.json"
SAFE  = BASE_DIR / "samples" / "customer_safe.json"

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


# ---------- helpers ----------
def load_json(path: Path) -> dict:
    """Load JSON, tolerating files saved with a UTF-8 BOM."""
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def post_json(base_url: str, endpoint: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------- UI ----------
st.sidebar.header("API base URL")
base = st.sidebar.text_input("API base URL", value=DEFAULT_API)

st.title("Customer Churn Predictor")
st.write("This dashboard sends payloads to the FastAPI service and shows the response.")


# Health check
if st.button("Check API health"):
    try:
        resp = requests.get(base.rstrip("/") + "/health", timeout=5)
        resp.raise_for_status()
        st.success(resp.json())
    except Exception as e:
        st.error(f"Health check failed: {e}")


st.subheader("Quick test with sample payloads")
col1, col2 = st.columns(2)

with col1:
    if st.button("Send risky sample"):
        try:
            payload = load_json(RISKY)
            st.json(post_json(base, "predict", payload))
        except Exception as e:
            st.error(e)

with col2:
    if st.button("Send safe sample"):
        try:
            payload = load_json(SAFE)
            st.json(post_json(base, "predict", payload))
        except Exception as e:
            st.error(e)


st.subheader("Or paste/edit a payload")

# Prefill editor with risky sample
try:
    prefill = json.dumps(load_json(RISKY), indent=2)
except Exception:
    prefill = "{}"

text = st.text_area("Payload (JSON)", value=prefill, height=380)

if st.button("Predict with JSON above"):
    try:
        # Be resilient if pasted text has a BOM
        clean = text.lstrip("\ufeff")
        payload = json.loads(clean)
        st.json(post_json(base, "predict", payload))
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
