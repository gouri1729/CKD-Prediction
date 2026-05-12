# main_rf.py

from preprocessing import load_data, preprocess_data, split_data
from evaluation import evaluate_model
from explainable import run_rf_shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

FILE = "CKD_MODIFIED.xlsx"
TARGET = "Target"


def main():

    # Load dataset
    df = load_data(FILE)

    # Preprocess
    df = preprocess_data(df, TARGET)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, TARGET)

    rfe = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=15
    )

    model = Pipeline([
        ("feature_selection", rfe),

        ("smote_tomek", SMOTETomek(
            sampling_strategy=0.8,
            random_state=42
        )),

        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        ))
    ])


    # # Build model
    # model = build_rf_tomek_pipeline()

    # Train
    model.fit(X_train, y_train)

    # Validation evaluation
    val_results = evaluate_model(model, X_val, y_val)

    print("\nValidation Results")
    print(val_results)

    # Test evaluation
    test_results = evaluate_model(model, X_test, y_test)

    print("\nTest Results")
    print(test_results)

    # Explainability
    run_rf_shap(model, X_train)


if __name__ == "__main__":
    main()