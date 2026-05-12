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

import joblib
import json


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

    # Fix inconsistent labels
    df['Target'] = df['Target'].replace({
        'High risk': 'High Risk',
        'High-risk': 'High Risk'
    })

    # 🔹 UPDATED LABEL MAPPING (3 CLASS)
    label_mapping = {
        "No Disease": 0,
        "Low Risk": 1,
        "Moderate Risk": 1,
        "High Risk": 1,
        "Severe Disease": 2
    }

    df["Target"] = df["Target"].map(label_mapping)

    # Ordinal mappings
    ordinal_mappings = {

        'Appetite (good/poor)': {
            'poor': 0,
            'good': 1
        },

        'Physical activity level': {
            'low': 0,
            'moderate': 1,
            'high': 2
        }
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Binary mappings
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

    # One-hot encoding
    nominal_cols = ['Smoking status']
    existing_nominal = [c for c in nominal_cols if c in df.columns]

    df = pd.get_dummies(df, columns=existing_nominal, drop_first=True)

    return df


# =====================================
# 4️⃣ Load and Preprocess Dataset
# =====================================

# df = load_data("CKD_MODIFIED.xlsx")
df = load_data("latest_ckd.xlsx")
df = preprocess_data(df)

target_column = "Target"

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df[target_column].value_counts())


# =====================================
# 5️⃣ Separate Features and Target
# =====================================

X = df.drop(target_column, axis=1)
y = df[target_column]


# =====================================
# 6️⃣ Identify Column Types
# =====================================

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64','float64']).columns


# =====================================
# 7️⃣ Preprocessing Pipelines
# =====================================

num_pipeline = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ]
)


# =====================================
# 8️⃣ Train / Validation / Test Split
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
# 9️⃣ Feature Selection (RFE)
# =====================================

rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=10
)


# =====================================
# 🔟 Full Pipeline
# =====================================

model = Pipeline(steps=[

    ('preprocessing', preprocessor),

    ('feature_selection', rfe),

    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),

    ('cluster', ClusterCentroids(random_state=42)),

    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ))
])


# =====================================
# 1️⃣1️⃣ Train Model
# =====================================

model.fit(X_train, y_train)


# =====================================
# 1️⃣2️⃣ Validation Evaluation
# =====================================

y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("\nValidation ROC-AUC Score:",
      roc_auc_score(y_val, y_val_prob, multi_class='ovr'))


# =====================================
# 1️⃣3️⃣ Test Prediction
# =====================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


# =====================================
# 1️⃣4️⃣ Final Test Evaluation
# =====================================

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTest ROC-AUC Score:",
      roc_auc_score(y_test, y_prob, multi_class='ovr'))


# =====================================
# 1️⃣5️⃣ Save Model + Metadata
# =====================================

feature_medians = {col: float(X[col].median()) for col in X.columns}

reverse_label_mapping = {
    0: "No Disease/Healthy",
    1: "Risky",
    2: "Has Disease"
}

metadata = {
    "features": list(X.columns),
    "feature_medians": feature_medians,
    "reverse_label_mapping": reverse_label_mapping,
    "num_classes": 3
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

joblib.dump(model, "ckd_model.pkl")

print("\nModel saved → ckd_model.pkl")
print("Metadata saved → model_metadata.json")

# Export X_train for app.py
__all__ = ['X_train']