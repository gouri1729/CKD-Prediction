# =====================================
# 1️⃣ Import Libraries
# =====================================

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline

import shap
import matplotlib.pyplot as plt


# =====================================
# 2️⃣ Data Loading Function
# =====================================

def load_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df


# =====================================
# 3️⃣ Preprocessing Function
# =====================================

def preprocess_data(df):

    df['Target'] = df['Target'].replace({
        'High risk': 'High Risk',
        'High-risk': 'High Risk'
    })

    label_mapping = {
        "No Disease": 0,
        "Low Risk": 1,
        "Moderate Risk": 1,
        "High Risk": 1,
        "Severe Disease": 1
    }

    df["Target"] = df["Target"].map(label_mapping)

    ordinal_mappings = {
        'Appetite (good/poor)': {'poor': 0, 'good': 1},
        'Physical activity level': {'low': 0, 'moderate': 1, 'high': 2}
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    binary_cols = {
        'Red blood cells in urine': {'normal': 0, 'abnormal': 1},
        'Pus cells in urine': {'normal': 0, 'abnormal': 1},
        'Pus cell clumps in urine': {'not present': 0, 'present': 1},
        'Bacteria in urine': {'not present': 0, 'present': 1},
        'Hypertension (yes/no)': {'no': 0, 'yes': 1},
        'Diabetes mellitus (yes/no)': {'no': 0, 'yes': 1},
        'Coronary artery disease (yes/no)': {'no': 0, 'yes': 1},
        'Pedal edema (yes/no)': {'no': 0, 'yes': 1},
        'Anemia (yes/no)': {'no': 0, 'yes': 1},
        'Family history of chronic kidney disease': {'no': 0, 'yes': 1},
        'Urinary sediment microscopy results': {'normal': 0, 'abnormal': 1}
    }

    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=['Smoking status'], drop_first=True)

    return df


# =====================================
# 4️⃣ Load Dataset
# =====================================

df = load_data("../data/CKD_MODIFIED.xlsx")
df = preprocess_data(df)

target_column = "Target"

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df[target_column].value_counts())


# =====================================
# 5️⃣ Separate Features
# =====================================

X = df.drop(target_column, axis=1)
y = df[target_column]

ALL_FEATURES = X.columns.tolist()
print(f"\nTotal Features: {len(ALL_FEATURES)}")


# =====================================
# 6️⃣ Train / Validation / Test Split
# =====================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

print("\nDataset Split:")
print("Training  :", len(X_train))
print("Validation:", len(X_val))
print("Testing   :", len(X_test))


# =====================================
# 7️⃣ Build Model Pipeline (ALL features)
# =====================================

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote',   SMOTE(random_state=42)),
    ('cluster', ClusterCentroids(random_state=42)),
    ('rf',      RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    random_state=42
                ))
])


# =====================================
# 8️⃣ Train Model
# =====================================

print("\nTraining Model...")
model.fit(X_train, y_train)


# =====================================
# 9️⃣ Validation Evaluation
# =====================================

y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("\nValidation ROC-AUC:",
      roc_auc_score(y_val, y_val_prob[:, 1]))


# =====================================
# 🔟 Test Evaluation
# =====================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTest ROC-AUC:",
      roc_auc_score(y_test, y_prob[:, 1]))


# =====================================
# 1️⃣1️⃣ Save Model + Metadata
# =====================================

feature_medians = {col: float(X[col].median()) for col in ALL_FEATURES}

metadata = {
    "features": ALL_FEATURES,
    "feature_medians": feature_medians,
    "num_classes": 2
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

joblib.dump(model, "ckd_model.pkl")

print("\nModel saved → ckd_model.pkl")
print("Metadata saved → model_metadata.json")


# =====================================
# 1️⃣2️⃣ SHAP Explainability (All Features)
# =====================================

print("\nRunning SHAP Explainability...")

rf_model = model.named_steps["rf"]

# Pass X_train through imputer only (not SMOTE/ClusterCentroids)
X_train_imputed = model.named_steps["imputer"].transform(X_train)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)

explainer = shap.TreeExplainer(rf_model)
explanation = explainer(X_train_imputed)

# Binary classifier → shape is (n_samples, n_features, 2) → slice class 1
if len(explanation.values.shape) == 3:
    explanation_class1 = shap.Explanation(
        values        = explanation.values[:, :, 1],
        base_values   = explanation.base_values[:, 1],
        data          = explanation.data,
        feature_names = X_train.columns.tolist()
    )
else:
    explanation_class1 = explanation

# ── Beeswarm Plot ──────────────────────────────────────────
plt.figure(figsize=(16, 12))
shap.plots.beeswarm(explanation_class1, max_display=20, show=False)
plt.title("SHAP Beeswarm Plot – Top 20 Features", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → shap_beeswarm.png")

# ── Bar Plot ───────────────────────────────────────────────
plt.figure(figsize=(16, 12))
shap.plots.bar(explanation_class1, max_display=20, show=False)
plt.title("SHAP Feature Importance – Top 20 Features", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → shap_bar.png")