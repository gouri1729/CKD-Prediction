# evaluation.py

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results = {
        "Accuracy": report["accuracy"],
        "Precision_CKD": report["1"]["precision"],
        "Recall_CKD": report["1"]["recall"],
        "F1_CKD": report["1"]["f1-score"],
        "ROC_AUC": roc_auc
    }

    return results


def cross_validate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="roc_auc"
    )

    return np.mean(scores), np.std(scores)