# Modular Pipeline Folder

This folder contains a modular training pipeline for CKD prediction. Each file has a clear responsibility so the pipeline is easy to extend and maintain.

## Files and folders

- `pipeline_runner.py`
  - Main entry point that orchestrates loading data, building pipelines, running evaluation, and saving outputs.
  - Runs both imbalance strategies (cluster and tomek) in sequence.

- `data_utils.py`
  - Data loading and preprocessing utilities.
  - Keeps label mapping, ordinal encoding, and binary encoding logic.
  - Provides `split_data()` with stratified train/validation/test splits.

- `models.py`
  - Model registry for all supported estimators.
  - Add or remove models here to change what runs.

- `pipeline_utils.py`
  - Builds the imbalanced-learn pipeline with RFE and the selected imbalance strategy.
  - Keeps the pipeline structure consistent for every model.

- `evaluation_utils.py`
  - Cross-validation and evaluation helpers.
  - Prints selected features and calculates all metrics used in comparison.

- `io_utils.py`
  - Saves trained model artifacts and metadata into the output folder.

- `output/`
  - Contains generated `.pkl` model files, metadata JSON files, and comparison CSVs.

- `catboost_info/`
  - CatBoost training artifacts generated during model training.

- `CKD_MODIFIED.xlsx`
  - Input dataset used by the pipeline runner.

- `__init__.py`
  - Marks this folder as a Python package for clean imports.

- `__pycache__/`
  - Python bytecode cache directory (auto-generated).
