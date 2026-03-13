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

# Used by: `base.py`, `logistic_regression.py`
# Purpose: shared lasso-logistic regularization grid for tuning `C`
# in both raw and PCA logistic-regression workflows.
LASSO_GRID = np.logspace(-6, 4, 10)


# ---------------------------------------------------------------------------
# Used in: base_random_forest.py
# ---------------------------------------------------------------------------

# Used by: `base_random_forest.py`
# Purpose: fixed PCA variance-retention grid for baseline RF PCA runs,
# used to choose `reducer__n_components`.
BASE_RF_PCA_GRID = [0.85]

# Used by: `base_random_forest.py`, `random_forest.py`
# Purpose: shared fixed RF feature-subsampling candidates for tuning
# `classifier__max_features`.
RF_MAX_FEATURES_GRID = ['sqrt', 'log2', 0.5]

# Used by: `base_random_forest.py`
# Purpose: baseline RF hyperparameter grid for tuning tree depth and
# forest size in the non-PCA RF model, plus feature subsampling via
# `classifier__max_features`.
BASE_RF_PARAM_GRID = {
    'classifier__max_depth': [2, 4, 8, 16],
    'classifier__n_estimators': [100],
    'classifier__max_features': RF_MAX_FEATURES_GRID,
}

# Used by: `base_random_forest.py`
# Purpose: baseline RF-with-PCA hyperparameter grid for jointly tuning
# `reducer__n_components`, `classifier__max_depth`,
# `classifier__n_estimators`, and `classifier__max_features`.
PCA_RF_PARAM_GRID = {
    'reducer__n_components': BASE_RF_PCA_GRID,
    'classifier__max_depth': [2, 4, 8, 16],
    'classifier__n_estimators': [100],
    'classifier__max_features': RF_MAX_FEATURES_GRID,
}

# Used by: `base_random_forest.py`
# Purpose: baseline selector+RF grid for tuning the selector penalty
# strength (`feature_selector__estimator__C`) together with RF depth,
# forest size, and `classifier__max_features`.
SEL_RF_PARAM_GRID = {
    'feature_selector__estimator__C': [0.001, 0.01, 0.1],
    'classifier__max_depth': [2, 4, 8, 16],
    'classifier__n_estimators': [100],
    'classifier__max_features': RF_MAX_FEATURES_GRID,
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
