import pandas as pd
from data_utils import load_data, preprocess_data, split_data
from models import get_models
from pipeline_utils import build_pipeline
from evaluation_utils import (
    evaluate_model,
    cross_validate_model,
    print_selected_features
)
from io_utils import save_model


def run_pipeline(
    data_path: str,
    target_column: str,
    imbalance_strategy: str = "cluster",
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

    split = split_data(df, target_column, random_state=random_state)

    models = get_models(random_state=random_state)
    comparison_rows = []

    for name, base_model in models.items():
        print(f"\n=== Running Model: {name} ===")
        print(f"Imbalance strategy: {imbalance_strategy}")

        pipeline = build_pipeline(
            model=base_model,
            rfe_features=rfe_features,
            imbalance_strategy=imbalance_strategy,
            random_state=random_state
        )

        cv_metrics = cross_validate_model(
            pipeline,
            split.X_train,
            split.y_train,
            random_state=random_state
        )

        evaluation = evaluate_model(
            pipeline,
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            split.X_test,
            split.y_test
        )

        # print_selected_features(pipeline, list(split.X_train.columns))
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

        comparison_rows.append({
            "model": name,
            "imbalance_strategy": imbalance_strategy,
            **cv_metrics,
            **evaluation["metrics"]
        })

        if name == "RandomForest":
            save_model(
                model=pipeline,
                feature_columns=list(split.X_train.columns),
                feature_medians=feature_medians,
                output_dir=output_dir,
                output_prefix=f"{name.lower()}_{imbalance_strategy}_ckd_model",
                metadata_path=(
                    f"{name.lower()}_{imbalance_strategy}_model_metadata.json"
                )
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        f"{output_dir}/model_comparison_{imbalance_strategy}.csv",
        index=False
    )

    leaderboard = comparison_df.sort_values(
        by=["cv_roc_auc", "cv_f1"],
        ascending=False
    ).reset_index(drop=True)

    print(
        "\n=== Leaderboard (sorted by CV ROC-AUC, CV F1) ==="
    )
    print(leaderboard)


if __name__ == "__main__":
    for strategy in ["cluster", "tomek"]:
        run_pipeline(
            data_path="CKD_MODIFIED.xlsx",
            target_column="Target",
            imbalance_strategy=strategy,
            rfe_features=10,
            random_state=42,
            print_validation=False
        )
