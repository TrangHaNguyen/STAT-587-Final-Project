"""Shared grid definitions for model search."""

from __future__ import annotations


def choose_grid_variant(variant: str, left_values, center_values, right_values):
    """Select one of three grid presets by variant name."""
    options = {"left": left_values, "center": center_values, "right": right_values}
    if variant not in options:
        raise ValueError("GRID_VARIANT must be one of: left, center, right.")
    return options[variant]

BASELINE_PCA_GRID = [0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.99]

# Ordered as (left, center, right) to match choose_grid(...) usage.
LOGREG_PCA_GRID_OPTIONS = (
    [0.5, 0.55, 0.6, 0.65],
    [0.7, 0.75, 0.8, 0.85],
    [0.9, 0.925, 0.95, 0.99],
)

RF_PCA_GRID_OPTIONS = (
    [0.7, 0.8],
    [0.85, 0.9],
    [0.95, 0.99],
)

BASE_RF_PCA_GRID = [0.8, 0.9, 0.95]

BASE_RF_PARAM_GRID = {
    'classifier__max_depth': [2, 3, 5, 8, 15],
    'classifier__n_estimators': [50, 100, 200, 500],
}

PCA_RF_PARAM_GRID = {
    'reducer__n_components': BASE_RF_PCA_GRID,
    'classifier__max_depth': [2, 3, 5],
    'classifier__n_estimators': [250, 500],
}

SEL_RF_PARAM_GRID = {
    'feature_selector__estimator__C': [0.001, 0.01, 0.1],
    'classifier__max_depth': [2, 3, 5],
    'classifier__n_estimators': [500],
}

LOGREG_INTERNAL_C_GRID_OPTIONS = (
    [0.005, 0.01, 0.1, 1.0],
    [0.05, 0.1, 1.0, 10.0],
    [0.1, 1.0, 10.0, 100.0],
)

LOGREG_RIDGE_PCA_C_GRID_OPTIONS = (
    [0.001, 0.01, 0.1, 1.0],
    [0.01, 0.1, 1.0, 10.0],
    [0.1, 1.0, 10.0, 100.0],
)

DATA_CLEAN_LAG_GRID_OPTIONS = (
    [1, 2, [1, 2], [1, 2, 3]],
    [1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [2, 3], [1, 3]],
    [3, 4, 5, 6, 7, [2, 3], [3, 4], [2, 3, 4], [3, 5]],
)

DATA_CLEAN_LOOKBACK_GRID_OPTIONS = (
    [5, 7, 10, 12, 14, 17, 21],
    [7, 10, 14, 17, 21, 24, 28],
    [14, 17, 21, 24, 28, 32, 36],
)

DATA_CLEAN_CORR_THRESHOLD_OPTIONS = (
    [0.7, 0.8, 0.85],
    [0.8, 0.9, 0.95],
    [0.9, 0.95, 0.98],
)

RF_BASE_PARAM_GRID_OPTIONS = {
    "classifier__max_depth": (
        [1, 2, 3, 5],
        [2, 3, 5, 10],
        [3, 5, 10, 20],
    ),
    "classifier__n_estimators": (
        [100, 250],
        [250, 500],
        [500, 750],
    ),
}

RF_PCA_PARAM_GRID_OPTIONS = {
    "classifier__max_depth": (
        [1, 2, 3, 5],
        [2, 3, 5, 10],
        [3, 5, 10, 20],
    ),
    "classifier__n_estimators": (
        [100, 250],
        [250, 500],
        [500, 750],
    ),
}

RF_SELECTOR_C_GRID_OPTIONS = {
    "lasso": (
        [0.0001, 0.001, 0.01, 0.1],
        [0.001, 0.01, 0.1, 1],
        [0.01, 0.1, 1, 10],
    ),
    "ridge": (
        [0.0001, 0.001, 0.01, 0.1],
        [0.001, 0.01, 0.1, 1],
        [0.01, 0.1, 1, 10],
    ),
}

RF_SELECTOR_N_ESTIMATORS_OPTIONS = (
    [250],
    [500],
    [750],
)

SVM_LINEAR_C_GRID_OPTIONS = (
    [0.01, 0.1, 1],
    [0.1, 1, 10],
    [1, 10, 100],
)

SVM_GAMMA_GRID_OPTIONS = (
    ['scale', 'auto', 0.001, 0.01, 0.1],
    ['scale', 'auto', 0.01, 0.1, 1],
    ['scale', 'auto', 0.1, 1, 10],
)

SVM_DEGREE_GRID_OPTIONS = (
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
)
