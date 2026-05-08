"""
CKD Prediction Web App - Flask Backend
Uses ONLY the top 10 RFE-selected features
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import json
import io
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from lime import lime_tabular
from flask import Flask, request, jsonify, send_from_directory

from final_app.model import X_train

# ------------------------------------------------------------------
# Boot
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "..", "web")
WEB_DIR = os.path.abspath(WEB_DIR)

app = Flask(__name__, static_folder=WEB_DIR)

# Load model and metadata
model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))

with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
    META = json.load(f)

# Expected features (top 10)
FEATURES = META["features"]
MEDIANS  = META["feature_medians"]

# 3-class labels - convert string keys to integers
reverse_mapping = META.get("reverse_label_mapping", {
    "0": "No Disease Detected",
    "1": "Risk of CKD detected",
    "2": "CKD Detected"
})
LABEL_MAP = {int(k): v for k, v in reverse_mapping.items()}
NUM_CLASSES = META.get("num_classes", 3)

print("\n✅ Loaded model with", len(FEATURES), "features")
print(f"✅ 3-class model:")
for k, v in sorted(LABEL_MAP.items()):
    print(f"   {k}: {v}")

# ------------------------------------------------------------------
# Initialize LIME Explainer
# ------------------------------------------------------------------
# Load training data for LIME (use medians as a simple background)
training_data = X_train[FEATURES].values

# Simplified feature names for LIME
FEATURE_NAMES_SIMPLE = [f.split('(')[0].strip() for f in FEATURES]

# Class names list
CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]

# Initialize LIME explainer
try:
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=FEATURE_NAMES_SIMPLE,
        class_names=CLASS_NAMES,
        mode='classification',
        random_state=42
    )
    print("✅ LIME Explainer initialized")
except Exception as e:
    print(f"⚠️  LIME initialization warning: {e}")
    explainer = None

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
# LIME Explanation endpoint
# ------------------------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    """Generate LIME explanations for a prediction"""
    if explainer is None:
        return jsonify({"error": "LIME explainer not initialized"}), 500
        
    payload = request.get_json(force=True)

    # Build a row with median defaults
    row = {col: MEDIANS[col] for col in FEATURES}

    # Override with user-supplied values
    for ui_key, col_name in UI_TO_COL.items():
        if ui_key in payload:
            try:
                row[col_name] = float(payload[ui_key])
            except (TypeError, ValueError):
                pass

    # Create DataFrame in exact feature order
    df = pd.DataFrame([row])[FEATURES]
    instance = df.values[0]

    try:
        # Get predicted class
        pred_class = int(model.predict(df)[0])
        pred_label = LABEL_MAP.get(pred_class, f"Class {pred_class}")
        
        # Create a wrapper function for predict_proba that handles DataFrame input
        def predict_proba_wrapper(X):
            """Wrapper to ensure input is handled correctly"""
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=FEATURES)
            else:
                X_df = X
            return model.predict_proba(X_df)
        
        # Generate LIME explanation
        print(f"Generating LIME explanation for class {pred_class}...")
        exp = explainer.explain_instance(
            instance, 
            predict_proba_wrapper,
            num_features=10,
            top_labels=3
        )
        print("LIME explanation generated successfully")
        
        # Filter out low-impact features (only keep significant ones)
        lime_list = exp.as_list(label=pred_class)
        # Filter features with absolute weight > 0.001 (threshold for significance)
        significant_features = [(feat, weight) for feat, weight in lime_list if abs(weight) > 0.001]
        
        # Generate visualizations as base64 strings
        plots = {}
        
        # 1. Feature importance plot for predicted class (with filtered features)
        plots['feature_importance'] = generate_lime_feature_plot(significant_features, pred_class)
        
        # 2. Probability bar chart across all classes
        plots['probability_chart'] = generate_probability_chart(exp)
        
        # 3. Extract feature importance data (only significant features)
        feature_importance = []
        
        for feature_condition, weight in significant_features:
            # Parse feature name and value from condition
            feature_name = feature_condition.split('<=')[0].split('>')[0].strip()
            feature_importance.append({
                "feature": feature_name,
                "condition": feature_condition,
                "weight": float(weight),
                "impact": "Increases" if weight > 0 else "Decreases"
            })
        
        # Sort by absolute weight
        feature_importance.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        return jsonify({
            "plots": plots,
            "feature_importance": feature_importance,
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "explanation_type": "LIME"
        })
        
    except Exception as e:
        print(f"LIME Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def generate_lime_feature_plot(significant_features, pred_class):
    """Generate LIME feature importance plot as base64 string with custom styling"""
    try:
        # Extract feature names and weights
        features = []
        weights = []
        
        for feature_condition, weight in significant_features:
            # Shorten feature names for better display
            feature_name = feature_condition.split('<=')[0].split('>')[0].strip()
            features.append(feature_name)
            weights.append(weight)
        
        # Sort by absolute weight for better visualization
        sorted_indices = sorted(range(len(weights)), key=lambda i: abs(weights[i]))
        features = [features[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.5)), facecolor='#1e293b')
        
        # Create color gradient based on weight (positive = green, negative = red/orange)
        colors = []
        for w in weights:
            if w > 0:
                # Positive weights - shades of green
                intensity = min(abs(w) / max(abs(w) for w in weights) * 0.7 + 0.3, 1.0)
                colors.append((0.2, intensity, 0.3))  # Green gradient
            else:
                # Negative weights - shades of orange/red
                intensity = min(abs(w) / max(abs(w) for w in weights) * 0.7 + 0.3, 1.0)
                colors.append((intensity, 0.3, 0.2))  # Red/orange gradient
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(features)), weights, color=colors, 
                       edgecolor='#475569', linewidth=2, height=0.7)
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            width = bar.get_width()
            label_x = width + (0.001 if width > 0 else -0.001)
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{weight:.4f}',
                   ha=ha, va='center', color='#e2e8f0', 
                   fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_facecolor('#1e293b')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11, color='#e2e8f0', fontweight='500')
        ax.set_xlabel('Feature Weight (Impact on Prediction)', 
                     fontsize=12, color='#e2e8f0', fontweight='bold')
        ax.set_title(f'Feature Impact - {LABEL_MAP.get(pred_class, "Unknown")}', 
                    color='#e2e8f0', fontsize=16, fontweight='bold', pad=20)
        
        # Style x-axis
        ax.tick_params(axis='x', colors='#e2e8f0', labelsize=10)
        ax.tick_params(axis='y', colors='#e2e8f0', labelsize=11)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='#64748b', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.15, color='#475569', linestyle='--', linewidth=0.8)
        
        # Style spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#475569')
            ax.spines[spine].set_linewidth(2)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1e293b', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"LIME feature plot error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_probability_chart(exp):
    """Generate probability comparison chart as base64 string"""
    try:
        plt.figure(figsize=(10, 5), facecolor='#1e293b')
        
        # Get probabilities for all classes
        proba_dict = {}
        available_labels = exp.available_labels()
        
        for label in available_labels:
            proba = exp.predict_proba[label]
            class_name = LABEL_MAP.get(label, f"Class {label}")
            proba_dict[class_name] = proba * 100
        
        # Create bar chart
        classes = list(proba_dict.keys())
        probas = list(proba_dict.values())
        colors = ['#4ade80', '#fbbf24', '#f87171'][:len(classes)]
        
        bars = plt.bar(classes, probas, color=colors, edgecolor='#334155', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', color='#e2e8f0', fontsize=11, fontweight='bold')
        
        plt.ylabel('Probability (%)', fontsize=11, color='#e2e8f0')
        plt.title('Prediction Probabilities for All Classes', fontsize=13, color='#e2e8f0', pad=15)
        plt.ylim(0, 110)  # Extra space for labels
        
        # Style
        ax = plt.gca()
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        
        # Style x and y tick labels
        for label in ax.get_xticklabels():
            label.set_color('#e2e8f0')
        for label in ax.get_yticklabels():
            label.set_color('#e2e8f0')
            
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2, color='#334155', linestyle='--')
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1e293b', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Probability chart error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)