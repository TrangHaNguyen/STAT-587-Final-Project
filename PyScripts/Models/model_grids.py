"""Shared grid definitions for model search.

This file is the single source of truth for the active search grids used by:
- `PyScripts/Models/logistic_regression.py`
- `PyScripts/Models/random_forest.py`
- `PyScripts/Models/SVM.py`
- `PyScripts/Models/base_random_forest.py`

Each grid block below includes a short note showing which script(s) consume it.
"""

from __future__ import annotations


def choose_grid_variant(variant: str, left_values, center_values, right_values):
    """Select one of three grid presets by variant name."""
    options = {"left": left_values, "center": center_values, "right": right_values}
    if variant not in options:
        raise ValueError("GRID_VARIANT must be one of: left, center, right.")
    return options[variant]

# Used by: `base.py`
# Purpose: baseline PCA search grid for choosing the best `n_components`
# in the baseline comparison workflow. Kept fixed by design.
BASELINE_PCA_GRID = [0.65, 0.7,0.75, 0.8, 0.85, 0.90, 0.95, 0.99]


# Used by: `base_random_forest.py`
# Purpose: fixed PCA variance-retention grid for baseline RF PCA runs,
# used to choose `reducer__n_components`.
BASE_RF_PCA_GRID = [0.85]

# Used by: `base_random_forest.py`
# Purpose: baseline RF hyperparameter grid for tuning tree depth and
# forest size in the non-PCA RF model.
BASE_RF_PARAM_GRID = {
    'classifier__max_depth': [2, 3, 4, 5, 8, 15],
    'classifier__n_estimators': [20, 50, 100],
}



# Used by: `base_random_forest.py`
# Purpose: baseline RF-with-PCA hyperparameter grid for jointly tuning
# `reducer__n_components`, `classifier__max_depth`, and `classifier__n_estimators`.
PCA_RF_PARAM_GRID = {
    'reducer__n_components': BASE_RF_PCA_GRID,
    'classifier__max_depth': [2, 3, 4, 5, 8, 15],
    'classifier__n_estimators': [20, 50, 100],
}




# Used by: `base_random_forest.py`
# Purpose: baseline selector+RF grid for tuning the selector penalty
# strength (`feature_selector__estimator__C`) together with RF depth.
SEL_RF_PARAM_GRID = {
    'feature_selector__estimator__C': [0.0001,0.001, 0.01, 0.1, 1],
    'classifier__max_depth': [2, 3, 4, 5, 8, 15],
    'classifier__n_estimators': [50, 100, 250, 500],
}






# Ordered as (left, center, right) to match choose_grid(...) usage.
# Used by: `logistic_regression.py`
# Purpose: candidate PCA variance-retention levels used to tune
# `reducer__n_components` in logistic-regression PCA models.
LOGREG_PCA_GRID_OPTIONS = (
    [0.5, 0.55, 0.6, 0.65],
    [0.7, 0.75, 0.8, 0.85],
    [0.9, 0.925, 0.95, 0.99],
)


# Used by: `logistic_regression.py`
# Purpose: candidate `C` values passed into `LogisticRegressionCV`
# to tune the main logistic regularization strength.
LOGREG_INTERNAL_C_GRID_OPTIONS = (
    [0.005, 0.01, 0.1, 1.0],
    [0.05, 0.1, 1.0, 10.0],
    [0.1, 1.0, 10.0, 100.0],
)

# Used by: `logistic_regression.py`
# Purpose: candidate `classifier__C` values for ridge-style logistic
# models after PCA / feature-reduction steps.
LOGREG_RIDGE_PCA_C_GRID_OPTIONS = (
    [0.001, 0.01, 0.1, 1.0],
    [0.01, 0.1, 1.0, 10.0],
    [0.1, 1.0, 10.0, 100.0],
)

# Used by: `logistic_regression.py`, `random_forest.py`, `SVM.py`
# Purpose: candidate lag configurations for the data-cleaning /
# feature-engineering search (`lag_period`).
DATA_CLEAN_LAG_GRID_OPTIONS = (
    [1, 2, [1, 2], [1, 2, 3]],
    [1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [2, 3], [1, 3]],
    [3, 4, 5, 6, 7, [2, 3], [3, 4], [2, 3, 4], [3, 5]],
)

# Used by: `logistic_regression.py`, `random_forest.py`, `SVM.py`
# Purpose: candidate rolling-window lookback sizes for the data-cleaning /
# feature-engineering search (`lookback_period`).
DATA_CLEAN_LOOKBACK_GRID_OPTIONS = (
    [5, 7, 10, 12, 14, 17, 21],
    [7, 10, 14, 17, 21, 24, 28],
    [14, 17, 21, 24, 28, 32, 36],
)

# Used by: `logistic_regression.py`, `random_forest.py`, `SVM.py`
# Purpose: candidate correlation thresholds for pruning highly correlated
# features in the data-cleaning search (`corr_threshold`).
DATA_CLEAN_CORR_THRESHOLD_OPTIONS = (
    [0.7, 0.8, 0.85],
    [0.8, 0.9, 0.95],
    [0.9, 0.95, 0.98],
)

# Used by: `random_forest.py`
# Purpose: RF baseline search ranges for tuning `classifier__max_depth`
# and `classifier__n_estimators` in the main RF workflow.
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



# Used by: `random_forest.py`
# Purpose: candidate PCA variance-retention levels used to tune
# `reducer__n_components` in random-forest PCA models.
RF_PCA_GRID_OPTIONS = (
    [0.7, 0.8],
    [0.85, 0.9],
    [0.95, 0.99],
)

# Used by: `random_forest.py`
# Purpose: RF-with-PCA search ranges for tuning tree depth and forest size
# after `reducer__n_components` is searched from `RF_PCA_GRID_OPTIONS`.
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

# Used by: `random_forest.py`
# Purpose: selector penalty grids used to tune the logistic selector's
# `C` parameter for lasso- and ridge-based feature selection before RF fitting.
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

# Used by: `random_forest.py`
# Purpose: forest-size grid used in selector-based RF models after
# feature selection.
RF_SELECTOR_N_ESTIMATORS_OPTIONS = (
    [250],
    [500],
    [750],
)

# Used by: `SVM.py`
# Purpose: `C` grid for linear SVM tuning, and also reused as the base
# `C` search range in nonlinear SVM variants.
SVM_LINEAR_C_GRID_OPTIONS = (
    [0.01, 0.1, 1],
    [0.1, 1, 10],
    [1, 10, 100],
)

# Used by: `SVM.py`
# Purpose: `gamma` grid for RBF and polynomial SVM tuning.
SVM_GAMMA_GRID_OPTIONS = (
    ['scale', 'auto', 0.001, 0.01, 0.1],
    ['scale', 'auto', 0.01, 0.1, 1],
    ['scale', 'auto', 0.1, 1, 10],
)

# Used by: `SVM.py`
# Purpose: polynomial degree grid for polynomial-kernel SVM tuning.
SVM_DEGREE_GRID_OPTIONS = (
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
)
