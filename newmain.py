# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, cross_validate
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from preprocessing import load_data, preprocess_data, split_data
# from evaluation import evaluate_model

# from models import *

# file="ckd_mod.xls"
# df = load_data(file)

# df = preprocess_data(df)
# X_train, X_test, y_train, y_test = split_data(df, "Target")

# results_list = []

# rf_base = get_rf_baseline()
# rf_base.fit(X_train, y_train)
# rf_base_results = evaluate_model(rf_base, X_test, y_test)
# rf_base_results["Model"] = "RF_Baseline"
# results_list.append(rf_base_results)



# rf_weighted = get_rf_class_weighted()
# rf_weighted.fit(X_train, y_train)
# rf_weighted_results = evaluate_model(rf_weighted, X_test, y_test)
# rf_weighted_results["Model"] = "RF_ClassWeighted"
# results_list.append(rf_weighted_results)

# # model = get_class_weighted_model()
# # model.fit(X_train_smote, y_train_smote)
# # results=evaluate_model(model, X_test, y_test)

# print(y_train.value_counts())
# print(y_test.value_counts())
# results_df = pd.DataFrame(results_list)
# results_df = results_df.set_index("Model")

# print(results_df)


import pandas as pd
from preprocessing import load_data, preprocess_data, split_data
from evaluation import evaluate_model
from models import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file="CKD_MODIFIED.xlsx"
df = load_data(file)

df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df, "Target")

results_list = []

# ===============================
# Helper function to run models
# ===============================

def run_model(model, model_name):

    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    results["Model"] = model_name

    results_list.append(results)

    # Predictions for confusion matrix
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion Matrix for {model_name}:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease","CKD"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


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

print("\nModel Performance Summary:")
print(results_df)