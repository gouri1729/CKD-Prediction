import os
import pandas as pd
import matplotlib
from preprocessing import load_data, preprocess_data, split_data_with_test
from evaluation import evaluate_model
from models import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate

import matplotlib.pyplot as plt

matplotlib.use("Agg")

file = "CKD_MODIFIED.xlsx"

df = load_data(file)
df = preprocess_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data_with_test(df, "Target")

results_list = []

# ===============================
# Cross Validation Setup
# ===============================

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

# ===============================
# Helper function to run models
# ===============================

def run_model(model, model_name):

    # ===============================
    # Cross Validation
    # ===============================

    cv_results = cross_validate(
        model,
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

    # ===============================
    # Train Final Model
    # ===============================

    model.fit(X_train, y_train)

    # ===============================
    # Test Evaluation
    # ===============================

    results = evaluate_model(model, X_test, y_test)

    results["Model"] = model_name

    # Add CV metrics to results table
    results["CV_Accuracy"] = cv_results["test_accuracy"].mean()
    results["CV_Precision"] = cv_results["test_precision"].mean()
    results["CV_Recall"] = cv_results["test_recall"].mean()
    results["CV_F1"] = cv_results["test_f1"].mean()
    results["CV_ROC_AUC"] = cv_results["test_roc_auc"].mean()

    results_list.append(results)

    # ===============================
    # Confusion Matrix
    # ===============================

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion Matrix for {model_name}:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "CKD"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"newmain_cm_{model_name}.png"))
    plt.close()


# ===============================
# Random Forest
# ===============================

run_model(get_rf_baseline(), "RF_Baseline")
run_model(get_rf_class_weighted(), "RF_ClassWeighted")

# ===============================
# Logistic Regression
# ===============================

run_model(get_lr_baseline(), "LogReg_Baseline")
run_model(get_lr_class_weighted(), "LogReg_ClassWeighted")

# ===============================
# SVM
# ===============================

run_model(get_svm_baseline(), "SVM_Baseline")
run_model(get_svm_class_weighted(), "SVM_ClassWeighted")

# ===============================
# XGBoost
# ===============================

run_model(get_xgb_baseline(), "XGB_Baseline")
run_model(get_xgb_class_weighted(y_train), "XGB_ClassWeighted")

# ===============================
# CatBoost
# ===============================

run_model(get_catboost_baseline(), "CatBoost_Baseline")
run_model(get_catboost_class_weighted(y_train), "CatBoost_ClassWeighted")

# ===============================
# Dataset distribution
# ===============================

print("\nTrain Distribution:")
print(y_train.value_counts())

print("\nTest Distribution:")
print(y_test.value_counts())

# ===============================
# Results Table
# ===============================

results_df = pd.DataFrame(results_list)

results_df = results_df.set_index("Model")

# Optional sorting
results_df = results_df.sort_values(
    by="CV_F1",
    ascending=False
)

print("\nModel Performance Summary:")
print(results_df)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(
    os.path.join(output_dir, "newmain_results.csv")
)