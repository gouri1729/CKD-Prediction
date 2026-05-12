import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from preprocessing import load_data, preprocess_data, split_data_with_test
from models import get_models
from imbalance_methods import build_pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def map_multiclass_target(target_series: pd.Series) -> pd.Series:
    normalized = target_series.astype(str).str.strip().replace({
        "High risk": "High Risk",
        "High-risk": "High Risk"
    })

    mapping = {
        "No Disease": 0,
        "Low Risk": 1,
        "Moderate Risk": 1,
        "High Risk": 1,
        "Severe Disease": 2
    }

    return normalized.map(mapping)


def cross_validate_multiclass(model, X_train, y_train, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr"
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
        "cv_precision_macro": float(np.mean(cv_results["test_precision_macro"])),
        "cv_recall_macro": float(np.mean(cv_results["test_recall_macro"])),
        "cv_f1_macro": float(np.mean(cv_results["test_f1_macro"])),
        "cv_roc_auc_ovr": float(np.mean(cv_results["test_roc_auc_ovr"]))
    }


def main():
    df = load_data("CKD_MODIFIED.xlsx")

    # Preserve original labels for multiclass target mapping.
    raw_target = df["Target"].copy()

    df = preprocess_data(df)
    df["Target"] = map_multiclass_target(raw_target)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_with_test(
        df,
        "Target"
    )

    base_models = get_models(random_state=42)

    for strategy in ["cluster", "tomek"]:
        results_rows = []
        print(f"\n=== Multiclass Run: {strategy} ===")

        for name, base_model in base_models.items():
            print(f"\n=== Running Model: {name} (multiclass) ===")
            print(f"Imbalance strategy: {strategy}")

            model = base_model
            if name == "CatBoost":
                model = CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    verbose=0,
                    random_state=42
                )
            elif name == "XGBoost":
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=42
                )

            pipeline = build_pipeline(
                model=model,
                rfe_features=10,
                imbalance_strategy=strategy,
                random_state=42,
                tomek_sampling_strategy="auto"
            )

            cv_metrics = cross_validate_multiclass(
                pipeline,
                X_train,
                y_train,
                random_state=42
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)

            print("\nTest Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            print("\nTest Classification Report:")
            print(classification_report(y_test, y_pred))

            roc_auc = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr"
            )
            print("\nTest ROC-AUC (OVR):")
            print(roc_auc)

            results_rows.append({
                "model": name,
                "imbalance_strategy": strategy,
                **cv_metrics,
                "test_roc_auc_ovr": float(roc_auc)
            })

        results_df = pd.DataFrame(results_rows)
        print("\n=== Multiclass CV Summary ===")
        print(results_df)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(
            os.path.join(
                output_dir,
                f"multiclass_results_{strategy}.csv"
            ),
            index=False
        )


if __name__ == "__main__":
    main()
