import json, argparse
from pathlib import Path
import pandas as pd
import joblib

# project root = one level above /src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

def load_threshold(default=0.5):
    try:
        return json.loads(THRESHOLD_PATH.read_text())["threshold"]
    except Exception:
        return default

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to JSON with raw customer fields")
    p.add_argument("--out", "-o", help="(optional) Save result JSON here")
    args = p.parse_args(argv)

    pipe = joblib.load(MODEL_PATH)
    threshold = load_threshold()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    X = pd.DataFrame([payload])  # one-row frame with raw features

    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)

    result = {"threshold": threshold, "prob_churn": round(proba, 4), "churn_pred": pred}
    print(json.dumps(result, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
