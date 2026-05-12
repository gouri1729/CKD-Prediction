from runner_utils import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        data_path="CKD_MODIFIED.xlsx",
        target_column="Target",
        imbalance_strategy="cluster",
        rfe_features=10,
        random_state=42,
        print_validation=False,
        output_dir="output"
    )