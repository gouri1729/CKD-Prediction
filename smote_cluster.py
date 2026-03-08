
# =====================================
# 1️⃣ Import Libraries
# =====================================


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline


from sklearn.feature_selection import RFE


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline




# =====================================
# 2️⃣ Load Dataset
# =====================================


df = pd.read_excel("CKD_MODIFIED.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip()


target_column = "Target"


print("Dataset Shape:", df.shape)




# =====================================
# 3️⃣ Clean Target Labels
# =====================================


df[target_column] = df[target_column].astype(str).str.strip()


df[target_column] = df[target_column].replace({
    "High risk": "High Risk",
    "High-risk": "High Risk"
})
print(df[target_column].unique())



label_mapping = {
    "No Disease": 0,
    "Low Risk": 1,
    "Moderate Risk": 2,
    "High Risk": 3,
    "Severe Disease": 4
}

df[target_column] = df[target_column].map(label_mapping)

print("\nTarget Distribution:")
print(df[target_column].value_counts())




# =====================================
# 4️⃣ Select ONLY Top 10 RFE Features
# =====================================

# These are the top 10 features identified by RFE
TOP_10_FEATURES = [
    "Age of the patient",
    "Blood pressure (mm/Hg)",
    "Random blood glucose level (mg/dl)",
    "Blood urea (mg/dl)",
    "Serum creatinine (mg/dl)",
    "Sodium level (mEq/L)",
    "Potassium level (mEq/L)",
    "Hemoglobin level (gms)",
    "Estimated Glomerular Filtration Rate (eGFR)",
    "Urine protein-to-creatinine ratio"
]

# Keep only these features
X = df[TOP_10_FEATURES].copy()
y = df[target_column]

print(f"\n✅ Using only {len(TOP_10_FEATURES)} features:")
for i, feat in enumerate(TOP_10_FEATURES, 1):
    print(f"  {i}. {feat}")




# =====================================
# 5️⃣ Preprocessing (Numerical Only)
# =====================================

# All top 10 features are numerical - just use SimpleImputer directly
preprocessor = SimpleImputer(strategy='median')




# =====================================
# 6️⃣ Train / Validation / Test Split
# =====================================


X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.125,
    stratify=y_temp,
    random_state=42
)


print("\nDataset Split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# =====================================
# 7️⃣ Build Pipeline (No RFE - features pre-selected)
# =====================================

model = Pipeline(steps=[

    ('preprocessing', preprocessor),

    # Increase minority class
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),

    # Cluster-based undersampling
    ('cluster', ClusterCentroids(random_state=42)),

    # Final Classifier
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])




# =====================================
# 8️⃣ Train Model
# =====================================


model.fit(X_train, y_train)




# =====================================
# 9️⃣ Validation Evaluation
# =====================================


y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:,1]


print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))


print("\nValidation ROC-AUC Score:", roc_auc_score(y_val, y_val_prob))


# =====================================
# 🔟 Test Evaluation
# =====================================


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]


print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTest ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# =====================================
# 1️⃣1️⃣ Save Model + Metadata
# =====================================

import joblib
import json

# Compute medians for the top 10 features
feature_medians = {col: float(X[col].median()) for col in TOP_10_FEATURES}

metadata = {
    "features": TOP_10_FEATURES,
    "feature_medians": feature_medians,
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

joblib.dump(model, "ckd_model.pkl")

print("\n" + "="*50)
print("✅ Model saved  →  ckd_model.pkl")
print("✅ Metadata saved →  model_metadata.json")
print("="*50)
