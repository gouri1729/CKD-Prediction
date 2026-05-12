# Project Modular Pipeline (Existing Files)

This folder now reuses the existing project modules instead of keeping separate modular_* files. The pipeline logic is integrated directly into the standard project files while keeping Model_analysis/modular_pipeline unchanged.

## Updated files

- `preprocessing.py`
  - Uses the full label mapping, ordinal encoding, and binary encoding.
  - Keeps the original `split_data()` API and adds `split_data_with_test()` for train/val/test splitting.

- `models.py`
  - Adds `get_models()` registry for RandomForest, LogisticRegression, XGBoost, CatBoost, SVM, DecisionTree, and KNN.
  - Removes deprecated XGBoost parameters in the baseline helpers.

- `imbalance_methods.py`
  - Adds `build_pipeline()` with RFE + SMOTE + ClusterCentroids or SMOTETomek.
  - Keeps `apply_smote()` unchanged.

- `evaluation.py`
  - Adds full evaluation utilities: `evaluate_model_full()`, `cross_validate_metrics()`, and `print_selected_features()`.
  - Adds `save_model()` for writing model artifacts and metadata to `output/`.
  - Keeps the original `evaluate_model()` and `cross_validate_model()` for older scripts.

- `runner_utils.py`
  - Orchestrates the pipeline using the existing project files above.
  - Saves `.pkl` and metadata only for RandomForest.

- `main_cluster.py`
  - Entry point for RFE + SMOTE + ClusterCentroids.

- `main_tomek.py`
  - Entry point for RFE + SMOTETomek.

## Outputs

All generated files go into `project/output` and are replaced each run:
- RandomForest model: `randomforest_cluster_ckd_model.pkl` or `randomforest_tomek_ckd_model.pkl`
- Metadata JSON: `randomforest_cluster_model_metadata.json` or `randomforest_tomek_model_metadata.json`
- Comparison CSV: `model_comparison_cluster.csv` or `model_comparison_tomek.csv`

## How to run

- Cluster (RFE + SMOTE + ClusterCentroids):
  - `python main_cluster.py`

- Tomek (RFE + SMOTETomek):
  - `python main_tomek.py`
