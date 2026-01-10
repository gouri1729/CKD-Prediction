import pandas as pd
import numpy as np

df = pd.read_excel("CKD_dataset.xls")
# print(df.columns)

df['Target'] = df['Target'].replace(
    {
        'High risk':'High Risk',
        'High-risk':'High Risk'
    }
)
print(df['Target'].unique())

# No null values present
# print(df.isnull().values.any())
# print("/n null values per column: ",df.isnull().sum())

num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# print(cat_cols)

# #categorical to numerical conversion
# from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Manual Encoding
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
    # 1 → Any Risk
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

print("Final dataset shape:", df.shape)
print(df.head())

# -------------------------------
# Features and Target
# -------------------------------

X = df.drop('Target', axis=1)
Y = df['Target']

print("\nOverall class distribution:")
print(pd.Series(Y).value_counts())

# -------------------------------
# K-Fold Cross Validation
# -------------------------------

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

fold = 1

for train_index, val_index in skf.split(X, Y):
    print(f"\n========== Fold {fold} ==========")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

    # -------------------------------
    # Compute Class Weights
    # -------------------------------

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))
    print("Class weights:", class_weights)

    # Convert class weights to sample weights (XGBoost format)
    sample_weights = y_train.map(class_weights)

    # -------------------------------
    # XGBoost Model (Binary)
    # -------------------------------

    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    # Predict probabilities
    y_prob = xgb_model.predict_proba(X_val)[:, 1]

    # Threshold (can tune later)
    threshold = 0.30
    y_pred = (y_prob >= threshold).astype(int)

    # Store fold results
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    fold += 1

# -------------------------------
# Final Evaluation (All folds combined)
# -------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n========== FINAL K-FOLD RESULTS (Binary) ==========")
print("Accuracy:", accuracy_score(all_y_true, all_y_pred))
print("\nClassification Report:\n", classification_report(all_y_true, all_y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(all_y_true, all_y_pred))
