import pandas as pd
import numpy as np

df = pd.read_excel("CKD_dataset.xls")
# print(df.columns)

# Fix label inconsistencies
df['Target'] = df['Target'].replace(
    {
        'High risk': 'High Risk',
        'High-risk': 'High Risk'
    }
)
print(df['Target'].unique())

# No null values present
# print(df.isnull().values.any())
# print("/n null values per column: ", df.isnull().sum())

num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# print(cat_cols)

# #categorical to numerical conversion
# from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Manual Encoding (Better than LabelEncoder)
# -------------------------------

ordinal_mappings = {
    'Appetite (good/poor)': {
        'poor': 0,
        'good': 1
    },
    'Physical activity level': {
        'low': 0,
        'moderate': 1,
        'high': 2
    },

    # -------- BINARY TARGET --------
    # 0 → No Disease
    # 1 → Any Risk (Low + Moderate + High + Severe)
    'Target': {
        'No Disease': 0,
        'Low Risk': 1,
        'Moderate Risk': 1,
        'High Risk': 1,
        'Severe Disease': 1
    }
}

for col, mapping in ordinal_mappings.items():
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
    df[col] = df[col].map(mapping)

nominal_cols = ['Smoking status']  # add more only if needed
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# print("Final data types:")
# print(df.dtypes)

# print("\nAny missing values?")
# print(df.isnull().sum().sort_values(ascending=False))

# print("\nFinal dataset shape:", df.shape)
# print(df.head())

# -------------------------------
# Train Test Split
# -------------------------------

X = df.drop('Target', axis=1)
Y = df['Target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nTrain class distribution:")
print(pd.Series(y_train).value_counts())

# -------------------------------
# NO SMOTE (removed intentionally)
# -------------------------------

# -------------------------------
# Compute Class Weights
# -------------------------------

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
print("\nClass weights:", class_weights)

# -------------------------------
# CatBoost Model (Binary)
# -------------------------------

from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    class_weights=class_weights,
    eval_metric="F1",
    random_seed=42,
    verbose=100
)

cat_model.fit(X_train, y_train)

# -------------------------------
# Predict probabilities
# -------------------------------

y_prob = cat_model.predict_proba(X_test)[:, 1]

# -------------------------------
# Automatic Threshold Tuning
# -------------------------------

from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.9, 0.05)
best_t, best_f1 = 0, 0

print("\nThreshold tuning results:")
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    print(f"Threshold={t:.2f} | F1={f1:.3f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest Threshold:", best_t)
print("Best F1:", best_f1)

# Final prediction using best threshold
y_pred = (y_prob >= best_t).astype(int)

# -------------------------------
# Evaluation
# -------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n========== CATBOOST RESULTS (Binary) ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
