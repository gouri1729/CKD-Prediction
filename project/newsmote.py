import os
import pandas as pd
from preprocessing import load_data, preprocess_data, split_data_with_test
from evaluation import evaluate_model
from models import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

file = "CKD_MODIFIED.xlsx"

df = load_data(file)
df = preprocess_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data_with_test(df, "Target")

results_list = []


# =====================================
# Cross Validation Setup
# =====================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# =====================================
# Helper Function
# =====================================

def run_smote_model(model, model_name):

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    # =====================================
    # Cross Validation
    # =====================================

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring_metrics,
        return_train_score=False
    )

    print(f"\n===== Cross Validation : {model_name} =====")

    print("CV Accuracy:",
          cv_results["test_accuracy"].mean())

    print("CV Precision:",
          cv_results["test_precision"].mean())

    print("CV Recall:",
          cv_results["test_recall"].mean())

    print("CV F1:",
          cv_results["test_f1"].mean())

    print("CV ROC-AUC:",
          cv_results["test_roc_auc"].mean())

    # =====================================
    # Train Final Model
    # =====================================

    pipeline.fit(X_train, y_train)

    # =====================================
    # Test Prediction
    # =====================================

    y_pred = pipeline.predict(X_test)

    results = evaluate_model(pipeline, X_test, y_test)

    results["Model"] = model_name

    # Add CV metrics to results table
    results["CV_Accuracy"] = cv_results["test_accuracy"].mean()
    results["CV_Precision"] = cv_results["test_precision"].mean()
    results["CV_Recall"] = cv_results["test_recall"].mean()
    results["CV_F1"] = cv_results["test_f1"].mean()
    results["CV_ROC_AUC"] = cv_results["test_roc_auc"].mean()

    results_list.append(results)

    # =====================================
    # Confusion Matrix
    # =====================================

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion Matrix for {model_name}")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "CKD"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


# =====================================
# Run Models
# =====================================

run_smote_model(get_rf_baseline(), "RF_SMOTE")

run_smote_model(get_lr_baseline(), "LogReg_SMOTE")

run_smote_model(get_svm_baseline(), "SVM_SMOTE")

run_smote_model(get_xgb_baseline(), "XGB_SMOTE")

run_smote_model(get_catboost_baseline(), "CatBoost_SMOTE")


# =====================================
# Dataset Distribution
# =====================================

print("\nTrain Distribution")
print(y_train.value_counts())

print("\nTest Distribution")
print(y_test.value_counts())


# =====================================
# Final Comparison Table
# =====================================

results_df = pd.DataFrame(results_list)

results_df = results_df.set_index("Model")

print("\nModel Performance Comparison")
print(results_df)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(
    os.path.join(output_dir, "newsmote_results.csv")
)