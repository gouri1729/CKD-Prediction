"""
CKD Prediction Web App - Flask Backend
Uses ONLY the top 10 RFE-selected features
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ------------------------------------------------------------------
# Boot
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR  = os.path.join(BASE_DIR, "web")

app = Flask(__name__, static_folder=WEB_DIR)

# Load model and metadata
model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))

with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
    META = json.load(f)

# Expected features (top 10)
FEATURES = META["features"]
MEDIANS  = META["feature_medians"]

# 3-class labels
LABEL_MAP = META.get("reverse_label_mapping", {
    0: "No Disease/Healthy",
    1: "Risky",
    2: "Has Disease"
})
NUM_CLASSES = META.get("num_classes", 3)

print("\n✅ Loaded model with", len(FEATURES), "features")
print(f"✅ 3-class model:")
for k, v in sorted(LABEL_MAP.items()):
    print(f"   {k}: {v}")

# ------------------------------------------------------------------
# Mapping: UI field name → exact column name in the dataset
# ------------------------------------------------------------------
UI_TO_COL = {
    "age":            "Age of the patient",
    "blood_pressure": "Blood pressure (mm/Hg)",
    "glucose":        "Random blood glucose level (mg/dl)",
    "blood_urea":     "Blood urea (mg/dl)",
    "serum_creatinine": "Serum creatinine (mg/dl)",
    "sodium":         "Sodium level (mEq/L)",
    "potassium":      "Potassium level (mEq/L)",
    "hemoglobin":     "Hemoglobin level (gms)",
    "egfr":           "Estimated Glomerular Filtration Rate (eGFR)",
    "upcr":           "Urine protein-to-creatinine ratio",
}


# ------------------------------------------------------------------
# Routes — serve static web files
# ------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(WEB_DIR, "home.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    # Build a row with median defaults
    row = {col: MEDIANS[col] for col in FEATURES}

    # Override with user-supplied values
    for ui_key, col_name in UI_TO_COL.items():
        if ui_key in payload:
            try:
                row[col_name] = float(payload[ui_key])
            except (TypeError, ValueError):
                pass  # keep median

    # Create DataFrame in exact feature order
    df = pd.DataFrame([row])[FEATURES]

    try:
        pred_class = int(model.predict(df)[0])
        proba_all  = model.predict_proba(df)[0]  # All class probabilities
        pred_prob  = float(proba_all[pred_class])  # Probability of predicted class
        
        # Get prediction label
        pred_label = LABEL_MAP.get(pred_class, f"Class {pred_class}")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Risk assessment based on predicted class
    risk_level_map = {
        0: "Healthy",
        1: "At Risk",
        2: "Has Disease"
    }
    risk_level = risk_level_map.get(pred_class, "Unknown")

    # Build detailed probability breakdown
    prob_breakdown = {LABEL_MAP.get(i, f"Class {i}"): round(float(proba_all[i]) * 100, 1) 
                      for i in range(len(proba_all))}

    return jsonify({
        "prediction":       pred_label,
        "predicted_class":  pred_class,
        "probability":      round(pred_prob * 100, 1),
        "risk_level":       risk_level,
        "all_probabilities": prob_breakdown
    })


# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
