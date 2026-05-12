from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
            random_state=random_state
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_state=random_state
        )
        # "SVM": SVC(
        #     kernel="linear",
        #     probability=True,
        #     random_state=random_state
        # ),
        # "DecisionTree": DecisionTreeClassifier(
        #     max_depth=8,
        #     random_state=random_state
        # ),
        # "KNN": KNeighborsClassifier(
        #     n_neighbors=7
        # )
    }
