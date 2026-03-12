"""Shared grid definitions for model search.

Update PCA cutoff values here so the main model scripts stay consistent.
"""

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
