
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from preprocessing import load_data, preprocess_data, split_data

from models import *

file="ckd_mod.xls"
df = load_data(file)

df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df, "Target")
# X_train_smote, y_train_smote = apply_smote(X_train, y_train) 


# ===============================
# Setup
# ===============================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# ===============================
# Define All Models (Baseline Versions)
# ===============================

models = {
    "RF": get_rf_baseline(),
    "SVM": get_svm_baseline(),
    "LogReg": get_lr_baseline(),
    "XGB": get_xgb_baseline(),
    "CatBoost": get_catboost_baseline()
}

results = []

# ===============================
# Run CV + SMOTE for each model
# ===============================

for name, model in models.items():

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring_metrics,
        return_train_score=False
    )

    result_dict = {
        "Model": name,
        "Accuracy": cv_results["test_accuracy"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall": cv_results["test_recall"].mean(),
        "F1": cv_results["test_f1"].mean(),
        "ROC_AUC": cv_results["test_roc_auc"].mean()
    }

    results.append(result_dict)

# ===============================
# Convert to DataFrame
# ===============================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ROC_AUC", ascending=False)

print(results_df)

# ===============================
# Train Final Best Model (Logistic Regression)
# ===============================
# from sklearn.preprocessing import StandardScaler
# best_model = Pipeline([
#     ("smote", SMOTE(random_state=42)),
#     ("scaler", StandardScaler()),
#     ("model", get_lr_baseline())
# ])

# best_model.fit(X_train, y_train)

# # import pandas as pd
# lr_model = best_model.named_steps["model"]
# importance = pd.DataFrame({
#     "Feature": X_train.columns,
#     "Coefficient": lr_model.coef_[0]
# })

# importance["Abs_Coeff"] = importance["Coefficient"].abs()
# importance = importance.sort_values("Abs_Coeff", ascending=False)

# print(importance.head(10))


# ===============================
# Hyperparameter Tuning for Best Model (Logistic Regression)
# ===============================

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV

# pipeline = Pipeline([
#     ("smote", SMOTE(random_state=42)),
#     ("scaler", StandardScaler()),
#     ("model", get_lr_baseline())
# ])

# param_grid = {
#     "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
# }

# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=5,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# grid.fit(X_train, y_train)

# best_model = grid.best_estimator_

# print("Best parameters:", grid.best_params_)




# ===============================
# Explainable AI using SHAP
# ===============================

# import shap
# import numpy as np

# # Extract trained logistic regression model from pipeline
# lr_model = best_model.named_steps["model"]
# X_train_np = X_train.astype(float).values
# # SHAP explainer
# explainer = shap.LinearExplainer(lr_model, X_train_np)

# shap_values = explainer.shap_values(X_train_np)

# if isinstance(shap_values, list):
#     shap_values = shap_values[1]

# # Global feature importance
# shap.summary_plot(shap_values, X_train_np, feature_names=X_train.columns)
