"""Shared grid definitions for model search.

This file is the single source of truth for the active search grids used by:
- `PyScripts/Models/logistic_regression.py`
- `PyScripts/Models/random_forest.py`
- `PyScripts/Models/SVM.py`
- `PyScripts/Models/base_random_forest.py`

Each grid block below includes a short note showing which script(s) consume it.
"""

from __future__ import annotations

import numpy as np

# Single source of truth for time-series CV folds across the active model scripts.
TIME_SERIES_CV_SPLITS = 5

# Single source of truth for the final chronological train/test split.
TEST_SIZE = 0.2

# Keep chronological order in holdout splits for time-series modeling.
TRAIN_TEST_SHUFFLE = False


def choose_grid_variant(variant: str, left_values, center_values, right_values):
    """Select one of three grid presets by variant name."""
    options = {"left": left_values, "center": center_values, "right": right_values}
    if variant not in options:
        raise ValueError("GRID_VARIANT must be one of: left, center, right.")
    return options[variant]

# ---------------------------------------------------------------------------
# Used in: base.py
# ---------------------------------------------------------------------------

# Used by: `base.py`
# Purpose: baseline PCA search grid for choosing the best `n_components`
# in the baseline comparison workflow. Kept fixed by design.
BASELINE_PCA_GRID = [0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.99]

# Used by: `base.py`, `logistic_regression.py`
# Purpose: shared ridge-logistic regularization grid for tuning `C`
# in both raw and PCA logistic-regression workflows.
RIDGE_GRID = np.logspace(-6, 4, 10)

# Used by: `base.py`, `logistic_regression.py`, `random_forest.py`
# Purpose: shared solver tolerance for logistic-regression optimizers,
# including RF feature selectors built from logistic regression.
LOGISTIC_TOL = 1e-3

# Used by: `base.py`, `logistic_regression.py`, `random_forest.py`
# Purpose: shared iteration cap for all logistic-regression-based models.
LOGISTIC_MAX_ITER = 1000

# Used by: `base.py`, `logistic_regression.py`
# Purpose: baseline/no-regularization logistic solver chosen for fast,
# stable binary fits without regularization.
LOGISTIC_BASELINE_SOLVER = "lbfgs"

# Used by: `base.py`, `logistic_regression.py`
# Purpose: shared fast solver for ridge-style (L2) binary logistic models.
LOGISTIC_RIDGE_SOLVER = "liblinear"

# Used by: `base.py`, `logistic_regression.py`
# Purpose: shared fast solver for lasso-style (L1) binary logistic models.
LOGISTIC_LASSO_SOLVER = "liblinear"

# Used by: `base.py`, `logistic_regression.py`
# Purpose: shared lasso-logistic regularization grid for tuning `C`
# in both raw and PCA logistic-regression workflows.
LASSO_GRID = np.logspace(-6, 4, 10)


# ---------------------------------------------------------------------------
# Used in: base_random_forest.py
# ---------------------------------------------------------------------------

# Used by: `base_random_forest.py`
# Purpose: baseline RF hyperparameter grid for tuning tree depth and
# forest size in the non-PCA RF model, plus feature subsampling via
# `classifier__max_features`.
BASE_RF_PARAM_GRID = {
    'classifier__max_depth': [2, 4, 8, 16, 32, 64],
    'classifier__n_estimators': [100],
    'classifier__max_features': ['sqrt', 'log2', 0.5],
}

# Used by: `base_random_forest.py`
# Purpose: baseline RF-with-PCA hyperparameter grid for tuning tree depth,
# forest size, and `classifier__max_features` after PCA has already been
# selected in a separate stage.
PCA_RF_PARAM_GRID = {
    'classifier__max_depth': [2, 4, 8, 16, 32, 64],
    'classifier__n_estimators': [100],
    'classifier__max_features': ['sqrt', 'log2', 0.5],
}

# Used by: `base_random_forest.py`
# Purpose: baseline selector+RF grid for tuning the selector penalty
# strength (`feature_selector__estimator__C`) together with RF depth,
# forest size, and `classifier__max_features`.
SEL_RF_PARAM_GRID = {
    'feature_selector__estimator__C': [0.001, 0.01, 0.1],
    'classifier__max_depth': [2, 4, 8, 16, 32, 64],
    'classifier__n_estimators': [100],
    'classifier__max_features': ['sqrt', 'log2', 0.5],
}


# ---------------------------------------------------------------------------
# Used in: base_SVM.py
# ---------------------------------------------------------------------------

# Used by: `base_SVM.py`, `SVM.py`
# Purpose: fixed `C` grid for linear SVM tuning, and also reused as the
# base `C` search range in nonlinear SVM variants.
SVM_LINEAR_C_GRID_OPTIONS = [0.01, 0.1, 1, 10, 100]

# Used by: `base_SVM.py`, `SVM.py`
# Purpose: fixed `gamma` grid for RBF and polynomial SVM tuning.
SVM_GAMMA_GRID_OPTIONS = ['scale', 'auto', 0.01, 0.1, 1]

# Used by: `base_SVM.py`, `SVM.py`
# Purpose: fixed polynomial degree grid for polynomial-kernel SVM tuning.
SVM_DEGREE_GRID_OPTIONS = [2, 3, 4, 5, 6]

# Used by: `base_SVM.py`, `SVM.py`
# Purpose: shared solver tolerance for SVM training.
SVM_TOL = 5e-2


# ---------------------------------------------------------------------------
# Used in: SVM.py
# ---------------------------------------------------------------------------

# Used by: `SVM.py`, `base_SVM.py`
# Purpose: `C` grid for linear SVM tuning, and also reused as the base
# `C` search range in nonlinear SVM variants.
# Defined above in the `base_SVM.py` section because both scripts share it.

# Used by: `SVM.py`, `base_SVM.py`
# Purpose: `gamma` grid for RBF and polynomial SVM tuning.
# Defined above in the `base_SVM.py` section because both scripts share it.

# Used by: `SVM.py`, `base_SVM.py`
# Purpose: polynomial degree grid for polynomial-kernel SVM tuning.
# Defined above in the `base_SVM.py` section because both scripts share it.
