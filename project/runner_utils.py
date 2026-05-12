import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocessing import (
    load_data,
    preprocess_data,
    split_data_with_test
)
from models import get_models
from imbalance_methods import build_pipeline
from evaluation import (
    evaluate_model_full,
    cross_validate_metrics,
    print_selected_features,
    save_model
)


def run_pipeline(
    data_path: str,
    target_column: str,
    imbalance_strategy: str,
    rfe_features: int = 10,
    random_state: int = 42,
    print_validation: bool = False,
    output_dir: str = "output"
) -> None:
    df = load_data(data_path)
    df = preprocess_data(df)

    feature_df = df.drop(target_column, axis=1)
    feature_medians = {
        col: float(feature_df[col].median())
        for col in feature_df.columns
    }

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_with_test(
        df,
        target_column,
        random_state=random_state
    )

    models = get_models(random_state=random_state)
    comparison_rows = []
    results_list = []

    for name, base_model in models.items():
        print(f"\n=== Running Model: {name} ===")
        print(f"Imbalance strategy: {imbalance_strategy}")

        pipeline = build_pipeline(
            model=base_model,
            rfe_features=rfe_features,
            imbalance_strategy=imbalance_strategy,
            random_state=random_state
        )

        cv_metrics = cross_validate_metrics(
            pipeline,
            X_train,
            y_train,
            random_state=random_state
        )

        evaluation = evaluate_model_full(
            pipeline,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

        # print_selected_features(pipeline, list(X_train.columns))
        if print_validation:
            print("\nValidation Confusion Matrix:")
            print(evaluation["val_confusion_matrix"])
            print("\nValidation Classification Report:")
            print(evaluation["val_classification_report"])
            print("\nValidation ROC-AUC Score:")
            print(evaluation["val_roc_auc"])

        print("\nTest Confusion Matrix:")
        print(evaluation["test_confusion_matrix"])
        print("\nTest Classification Report:")
        print(evaluation["test_classification_report"])
        print("\nTest ROC-AUC Score:")
        print(evaluation["test_roc_auc"])

        cm = evaluation["test_confusion_matrix"]
        print(f"\nConfusion Matrix for {name}:")
        print(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Disease", "CKD"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

        results_list.append({
            "Model": name,
            "Accuracy": evaluation["metrics"]["test_accuracy"],
            "Precision_CKD": evaluation["metrics"]["test_precision"],
            "Recall_CKD": evaluation["metrics"]["test_recall"],
            "F1_CKD": evaluation["metrics"]["test_f1"],
            "ROC_AUC": evaluation["metrics"]["test_roc_auc"]
        })

        comparison_rows.append({
            "model": name,
            "imbalance_strategy": imbalance_strategy,
            **cv_metrics,
            **evaluation["metrics"]
        })

        if name == "RandomForest" and imbalance_strategy == "cluster":
            save_model(
                model=pipeline,
                feature_columns=list(X_train.columns),
                feature_medians=feature_medians,
                output_dir=output_dir,
                output_prefix=f"{name.lower()}_{imbalance_strategy}_ckd_model",
                metadata_path=(
                    f"{name.lower()}_{imbalance_strategy}_model_metadata.json"
                )
            )

    os.makedirs(output_dir, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        os.path.join(
            output_dir,
            f"model_comparison_{imbalance_strategy}.csv"
        ),
        index=False
    )

    leaderboard = comparison_df.sort_values(
        by=["cv_roc_auc", "cv_f1"],
        ascending=False
    ).reset_index(drop=True)

    print("\n=== Leaderboard (sorted by CV ROC-AUC, CV F1) ===")
    print(leaderboard)

    print("\nTrain Distribution:")
    print(y_train.value_counts())
    print("\nTest Distribution:")
    print(y_test.value_counts())

    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index("Model")
    print("\nModel Performance Comparison:")
    print(results_df)
