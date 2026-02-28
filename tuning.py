
from sklearn.model_selection import GridSearchCV

def tune_random_forest(model, X_train, y_train):

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_