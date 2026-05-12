# models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



# RANDOM FOREST

def get_rf_baseline():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )


def get_rf_class_weighted():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


# SVM

def get_svm_baseline():
    return SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )


def get_svm_class_weighted():
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )


# LOGISTIC REGRESSION

def get_lr_baseline():
    return LogisticRegression(
        max_iter=5000, #1000
        random_state=42
    )


def get_lr_class_weighted():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


# XGBOOST

def get_xgb_baseline():
    return XGBClassifier(
        n_estimators=200,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )


def get_xgb_class_weighted(y_train):
    # scale_pos_weight = negative_class / positive_class
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)

    scale_pos_weight = neg / pos

    return XGBClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )


# CATBOOST

def get_catboost_baseline():
    return CatBoostClassifier(
        iterations=200,
        verbose=0,
        random_state=42
    )


def get_catboost_class_weighted(y_train):
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)

    class_weights = [1, neg / pos]

    return CatBoostClassifier(
        iterations=200,
        class_weights=class_weights,
        verbose=0,
        random_state=42
    )

