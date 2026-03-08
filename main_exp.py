import pandas as pd
from preprocessing import load_data, preprocess_data, split_data
from models import *
from evaluation import evaluate_model
from imbalance_methods import apply_smote

file="CKD_changedataset.xlsx"
df = load_data(file)

df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df, "Target")
X_train_smote, y_train_smote = apply_smote(X_train, y_train) # apply smote on training data 


results_list = []

# rf_base = get_rf_baseline()
# rf_base.fit(X_train, y_train)
# rf_base_results = evaluate_model(rf_base, X_test, y_test)
# rf_base_results["Model"] = "RF_Baseline"
# results_list.append(rf_base_results)

from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_weighted = get_rf_class_weighted()

scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", get_rf_baseline())
])
cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring_metrics
)

print("CV Accuracy:", cv_results["test_accuracy"].mean())
print("CV Precision:", cv_results["test_precision"].mean())
print("CV Recall:", cv_results["test_recall"].mean())
print("CV F1:", cv_results["test_f1"].mean())
print("CV ROC-AUC:", cv_results["test_roc_auc"].mean())


pipeline.fit(X_train, y_train)

# ===============================
# Predictions on Test Set
# ===============================

y_pred = pipeline.predict(X_test)

# ===============================
# Confusion Matrix
# ===============================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()




# rf_weighted = get_rf_class_weighted()
# rf_weighted.fit(X_train, y_train)
# rf_weighted_results = evaluate_model(rf_weighted, X_test, y_test)
# rf_weighted_results["Model"] = "RF_ClassWeighted"
# results_list.append(rf_weighted_results)

# model = get_class_weighted_model()
# model.fit(X_train_smote, y_train_smote)
# results=evaluate_model(model, X_test, y_test)

# print(y_train.value_counts())
# print(y_test.value_counts())
# results_df = pd.DataFrame(results_list)
# results_df = results_df.set_index("Model")

# print(results_df)






#Flow of the code:
#1. Load and preprocess data
#2. Split data into training and testing sets
#3. Apply SMOTE to handle class imbalance on training data
#4. Train a class-weighted model (e.g., Random Forest)
#5. Evaluate the model on the test set
