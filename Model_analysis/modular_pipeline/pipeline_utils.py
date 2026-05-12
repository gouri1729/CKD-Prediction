from typing import Any

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline


def build_pipeline(
    model: Any,
    rfe_features: int = 10,
    imbalance_strategy: str = "cluster",
    random_state: int = 42
) -> Pipeline:
    # RFE uses a consistent estimator so any downstream model can be swapped in.
    rfe = RFE(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        n_features_to_select=rfe_features
    )

    if imbalance_strategy == "tomek":
        pipeline_steps = [
            ("feature_selection", rfe),
            ("smote_tomek", SMOTETomek(
                sampling_strategy=0.8,
                random_state=random_state
            )),
            ("model", model)
        ]
    else:
        pipeline_steps = [
            ("feature_selection", rfe),
            ("smote", SMOTE(sampling_strategy="auto", random_state=random_state)),
            ("cluster", ClusterCentroids(random_state=random_state)),
            ("model", model)
        ]

    return Pipeline(pipeline_steps)
