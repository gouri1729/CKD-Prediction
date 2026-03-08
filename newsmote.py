import pandas as pd
from preprocessing import load_data, preprocess_data, split_data
from evaluation import evaluate_model
from models import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

file="CKD_changedataset.xlsx"

df = load_data(file)
df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df, "Target")

results_list = []

# ===============================
# Helper function
# ===============================

def run_smote_model(model, model_name):

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    results = evaluate_model(pipeline, X_test, y_test)
    results["Model"] = model_name

    results_list.append(results)

    # Confusion matrix
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


# ===============================
# Models with SMOTE
# ===============================

run_smote_model(get_rf_baseline(), "RF_SMOTE")

run_smote_model(get_lr_baseline(), "LogReg_SMOTE")

run_smote_model(get_svm_baseline(), "SVM_SMOTE")

run_smote_model(get_xgb_baseline(), "XGB_SMOTE")

run_smote_model(get_catboost_baseline(), "CatBoost_SMOTE")


# ===============================
# Dataset distribution
# ===============================

print("\nTrain Distribution")
print(y_train.value_counts())

print("\nTest Distribution")
print(y_test.value_counts())


# ===============================
# Comparison Table
# ===============================

results_df = pd.DataFrame(results_list)
results_df = results_df.set_index("Model")

print("\nModel Performance Comparison")
print(results_df)