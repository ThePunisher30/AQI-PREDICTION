"""Evaluation metrics — MAE, RMSE, R² for model comparison."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def evaluate_all(y_true, y_pred):
    """Return dict of all metrics."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
