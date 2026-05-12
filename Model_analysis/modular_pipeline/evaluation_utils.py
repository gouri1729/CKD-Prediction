from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "val_precision": precision_score(y_val, y_val_pred),
        "val_recall": recall_score(y_val, y_val_pred),
        "val_f1": f1_score(y_val, y_val_pred),
        "val_roc_auc": roc_auc_score(y_val, y_val_prob),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_prob)
    }

    results = {
        "metrics": metrics,
        "val_confusion_matrix": confusion_matrix(y_val, y_val_pred),
        "val_classification_report": classification_report(y_val, y_val_pred),
        "val_roc_auc": metrics["val_roc_auc"],
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "test_classification_report": classification_report(y_test, y_test_pred),
        "test_roc_auc": metrics["test_roc_auc"]
    }

    return results


def cross_validate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
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

    return {
        "cv_accuracy": float(np.mean(cv_results["test_accuracy"])),
        "cv_precision": float(np.mean(cv_results["test_precision"])),
        "cv_recall": float(np.mean(cv_results["test_recall"])),
        "cv_f1": float(np.mean(cv_results["test_f1"])),
        "cv_roc_auc": float(np.mean(cv_results["test_roc_auc"]))
    }


def print_selected_features(model: Pipeline, feature_columns: List[str]) -> None:
    selector = model.named_steps["feature_selection"]
    selected_features = [
        feature_columns[i]
        for i, selected in enumerate(selector.support_)
        if selected
    ]
    print("\nSelected Features by RFE:")
    for i, feature in enumerate(selected_features, 1):
        print(i, feature)
