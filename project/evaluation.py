import json
import os
from typing import Dict, Any, List

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.3).astype(int)

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
    print("Unique predictions:", set(y_pred))
    print("True labels:", set(y_test))
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


def evaluate_model_full(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
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


def cross_validate_metrics(model, X_train, y_train, random_state=42) -> Dict[str, float]:
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


def print_selected_features(model, feature_columns: List[str]) -> None:
    selector = model.named_steps["feature_selection"]
    selected_features = [
        feature_columns[i]
        for i, selected in enumerate(selector.support_)
        if selected
    ]
    print("\nSelected Features by RFE:")
    for i, feature in enumerate(selected_features, 1):
        print(i, feature)


def save_model(
    model,
    feature_columns: List[str],
    feature_medians: Dict[str, float],
    output_dir: str,
    output_prefix: str,
    metadata_path: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    reverse_label_mapping = {
        0: "No Disease",
        1: "CKD/Risk"
    }

    selector = model.named_steps["feature_selection"]
    selected_features = [
        feature_columns[i]
        for i, selected in enumerate(selector.support_)
        if selected
    ]

    metadata = {
        "features": feature_columns,
        "feature_medians": feature_medians,
        "selected_features": selected_features,
        "reverse_label_mapping": reverse_label_mapping,
        "num_classes": 2
    }

    metadata_file = os.path.join(output_dir, metadata_path)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    model_file = os.path.join(output_dir, f"{output_prefix}.pkl")
    joblib.dump(model, model_file)