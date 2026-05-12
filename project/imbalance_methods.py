from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def build_pipeline(
    model,
    rfe_features=10,
    imbalance_strategy="cluster",
    random_state=42
):
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
            ("smote", SMOTE(
                sampling_strategy="auto",
                random_state=random_state
            )),
            ("cluster", ClusterCentroids(random_state=random_state)),
            ("model", model)
        ]

    return Pipeline(pipeline_steps)

