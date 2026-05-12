# explainability.py

import shap
import matplotlib.pyplot as plt


def run_rf_shap(model, X_train):

    print("\nRunning SHAP Explainability...")

    # Extract pipeline steps
    preprocessor = model.named_steps["preprocessing"]
    rfe = model.named_steps["feature_selection"]
    rf_model = model.named_steps["rf"]

    # Transform training data
    X_train_processed = preprocessor.transform(X_train)

    # Apply RFE mask
    X_train_selected = X_train_processed[:, rfe.support_]

    # Feature names
    feature_names = preprocessor.get_feature_names_out()
    selected_feature_names = feature_names[rfe.support_]

    # SHAP explainer
    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_train_selected)

    # Summary plot
    shap.summary_plot(
        shap_values[:, :, 1],
        X_train_selected,
        feature_names=selected_feature_names
    )

    # Bar importance
    shap.summary_plot(
        shap_values[:, :, 1],
        X_train_selected,
        feature_names=selected_feature_names,
        plot_type="bar"
    )

    # Force plot
    sample_index = 0

    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][sample_index],
        X_train_selected[sample_index],
        feature_names=selected_feature_names,
        matplotlib=True
    )