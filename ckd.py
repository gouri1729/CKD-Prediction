# CKD Prediction using UCI CKD Dataset
# ------------------------------------
# Models: Logistic Regression, Random Forest, XGBoost, SVM, KNN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_excel("CKD_dataset.xls")   # change if you have .xlsx

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ---------------------------
# Preprocessing
# ---------------------------

# Replace "?" with NaN (if present)
df.replace("?", np.nan, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Handle missing values
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype != "object" else col.fillna(col.mode()[0]))

# ---------------------------
# Features & Target
# ---------------------------
X = df.drop("Target", axis=1)
y = df["Target"]

# Encode target labels (CKD vs Not CKD)
y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Define Models
# ---------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

# ---------------------------
# Training & Evaluation
# ---------------------------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n============================")
    print(f" Model: {name}")
    print("============================")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Model Comparison
# ---------------------------
print("\nModel Comparison (Accuracy):")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
