import json
import os
from typing import Dict, List

import joblib
from imblearn.pipeline import Pipeline


def save_model(
    model: Pipeline,
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
