# Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.feature_selection import RFE

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

from catboost import CatBoostClassifier


# Data Loading Function

def load_data(file_path):

    df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()

    return df


# Preprocessing Function

def preprocess_data(df):

    # Fix inconsistent labels

    df['Target'] = df['Target'].replace({
        'High risk': 'High Risk',
        'High-risk': 'High Risk'
    })

    # Binary Classification Mapping

    label_mapping = {

        "No Disease": 0,

        "Low Risk": 1,

        "Moderate Risk": 1,

        "High Risk": 1,

        "Severe Disease": 1
    }

    df["Target"] = df["Target"].map(label_mapping)

    # Ordinal Encoding

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

    # Binary Encoding

    binary_cols = {

        'Red blood cells in urine': {
            'normal': 0,
            'abnormal': 1
        },

        'Pus cells in urine': {
            'normal': 0,
            'abnormal': 1
        },

        'Pus cell clumps in urine': {
            'not present': 0,
            'present': 1
        },

        'Bacteria in urine': {
            'not present': 0,
            'present': 1
        },

        'Hypertension (yes/no)': {
            'no': 0,
            'yes': 1
        },

        'Diabetes mellitus (yes/no)': {
            'no': 0,
            'yes': 1
        },

        'Coronary artery disease (yes/no)': {
            'no': 0,
            'yes': 1
        },

        'Pedal edema (yes/no)': {
            'no': 0,
            'yes': 1
        },

        'Anemia (yes/no)': {
            'no': 0,
            'yes': 1
        },

        'Family history of chronic kidney disease': {
            'no': 0,
            'yes': 1
        },

        'Urinary sediment microscopy results': {
            'normal': 0,
            'abnormal': 1
        }
    }

    for col, mapping in binary_cols.items():

        if col in df.columns:
            df[col] = df[col].map(mapping)

    # One Hot Encoding

    nominal_cols = ['Smoking status']

    existing_nominal = [
        c for c in nominal_cols
        if c in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=existing_nominal,
        drop_first=True
    )

    return df


# Load Dataset

df = load_data("CKD_MODIFIED.xlsx")

df = preprocess_data(df)

target_column = "Target"

print("Dataset Shape:", df.shape)

print("\nTarget Distribution:")
print(df[target_column].value_counts())


# Features and Target

X = df.drop(target_column, axis=1)

y = df[target_column]


# Train Validation Test Split

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


# Feature Selection using RFE

catboost_rfe = CatBoostClassifier(

    iterations=100,

    depth=6,

    learning_rate=0.1,

    verbose=0,

    random_state=42
)

rfe = RFE(

    estimator=catboost_rfe,

    n_features_to_select=20
)


# Final Pipeline

model = Pipeline([

    ('feature_selection', rfe),

    ('smote_tomek', SMOTETomek(

        sampling_strategy=0.8,

        random_state=42
    )),

    ('catboost', CatBoostClassifier(

        iterations=300,

        depth=6,

        learning_rate=0.05,

        loss_function='Logloss',

        eval_metric='AUC',

        verbose=0,

        random_state=42
    ))
])


# Cross Validation

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)

scoring = {

    'accuracy': 'accuracy',

    'precision': 'precision',

    'recall': 'recall',

    'f1': 'f1',

    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(

    estimator=model,

    X=X_train,

    y=y_train,

    cv=skf,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False
)

print("\nCross Validation Results")

print("\nAccuracy:",
      np.mean(cv_results['test_accuracy']))

print("Precision:",
      np.mean(cv_results['test_precision']))

print("Recall:",
      np.mean(cv_results['test_recall']))

print("F1 Score:",
      np.mean(cv_results['test_f1']))

print("ROC AUC:",
      np.mean(cv_results['test_roc_auc']))


# Train Model

model.fit(X_train, y_train)


# Validation Evaluation

y_val_pred = model.predict(X_val)

y_val_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation Confusion Matrix:")

print(confusion_matrix(y_val, y_val_pred))

print("\nValidation Classification Report:")

print(classification_report(y_val, y_val_pred))

print("\nValidation ROC-AUC Score:")

print(roc_auc_score(y_val, y_val_prob))


# Test Prediction

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# Final Test Evaluation

print("\nTest Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nTest Classification Report:")

print(classification_report(y_test, y_pred))

print("\nTest ROC-AUC Score:")

print(roc_auc_score(y_test, y_prob))


# Selected Features from RFE

feature_names = X.columns

rfe_mask = model.named_steps[
    'feature_selection'
].support_

selected_features = feature_names[rfe_mask]

print("\nSelected Features by RFE:")

for f in selected_features:
    print(f)


# Feature Ranking

ranking = model.named_steps[
    'feature_selection'
].ranking_

feature_ranking = pd.DataFrame({

    "Feature": feature_names,

    "Rank": ranking

}).sort_values(by="Rank")

print("\nFeature Ranking (RFE):")

print(feature_ranking)