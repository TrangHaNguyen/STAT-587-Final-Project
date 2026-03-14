## Project Background

This repository contains an individual version of a project that originally began as a group project. The work was later split into separate individual projects following a mutual decision by the contributors to proceed independently because of differences in objectives, methods, and findings.

The author contributed to the original upstream repository during the shared project stage, but made no further contributions to that repository after the split. From that point onward, this repository has been developed and maintained independently.

Some initial components in this repository originated from the earlier shared project stage, including parts of the data-loading code, model pipeline code, and exploratory data analysis code. The author acknowledges the other contributor’s earlier work on these components. In the present repository, these materials have been revised, extended, or adapted independently.

For reference, the original upstream repository from the earlier shared project stage is available here: [Original Upstream Repository] (https://github.com/Sabuchan123/STAT-587-Final-Project). Following the separation of the project, that repository continued as the other contributor’s independently maintained project, and the shared development stage ended on March 9, 2026.

## Instruction for Replication
(to be added later)
Work flow:
### 1. Set up the environment to work
This project requires Python 3.11 (TensorFlow compatibility) for reproducible setup because it use some packages in this version (those are not available in the most updated one on March 11th 2026)

#### On macOS
Install Python 3.11 with Homebrew from the terminal:
```bash
brew install python@3.11
```
From the project root, run:
```bash
PYTHON_BIN=python3.11 bash setup_env.sh
source .venv/bin/activate
```
This will create a local `.venv` and activate the environment.

Example entry points:
If you need to compile the report in `LaTex/`, install the LaTeX system packages noted in `requirements.txt`.
### Example after setting up the enviroment to run code
```bash
/opt/homebrew/bin/python3.11 --version
PYTHON_BIN=/opt/homebrew/bin/python3.11 bash setup_env.sh
source .venv/bin/activate
python --version
python PyScripts/Data/loading_daily_data.py
```

### 2. Downloading data. Code and results are infolder Data
- hourly data: loading_hourly_data.py
- 2 yrs and 8 yrs data: loading_daily_data.py. The output is raw_data_8_years.parquet.
Note: there is slightly differences in the data downloaded for replication on March 10th compared with the original data downloaded by the other contributor in this project.
They can be found in difference.csv and ticker_largest_difference. Potential explanation: different version of yfinance, different options in downloading(less likely as I tried multiple options)

### 3. Run codes for different part of the report
- EDA part: EDA_simple

### 4. Run codes for base model of only OHLCV features (and days of week for cyclical adjustment)
- `PyScripts/Models/base.py` saves fitted model caches and diagnostic caches in `PyScripts/cache/`.
- If you do not want to retrain the models, keep those `.pkl` files and inspect them later using the commented helper examples at the end of [base.py]
- The model `.pkl` files store fitted estimators; the diagnostics `.pkl` files store the cached bias-variance and train/test curve values used for the figures.

### Model file inventory
Naming convention used in the active model scripts:
- `Base` is reserved for logistic models without regularization.
- `Raw` refers to the no-DOW / no-additional-feature branch used in the raw-feature family scripts such as `base.py`, `base_SVM.py`, and `base_random_forest.py`.

`PyScripts/Models/base.py`
- `Base+PCA`
- `Raw Ridge`
- `Raw LASSO`
- `Raw Ridge+PCA`
- `Raw LASSO+PCA`
- `Base+DOW+PCA`
- `Ridge+DOW`
- `LASSO+DOW`
- `Ridge+PCA+DOW`
- `LASSO+PCA+DOW`

`PyScripts/Models/logistic_regression.py`
- `PCA Base`
- `Ridge Log. Reg.`
- `LASSO Log. Reg.`
- `PCA Ridge(int.) Log. Reg.`
- `PCA LASSO(int.) Log. Reg.`
- `Elastic Net Log. Reg.` is currently present but commented out

`PyScripts/Models/random_forest.py`
- `Base RF`
- `PCA RF`
- `LASSO RF`
- `Ridge RF`
- `Stepwise RF` exists only in a disabled block

`PyScripts/Models/SVM.py`
- `Linear SVM`
- `RBF SVM`
- `Poly SVM`

`PyScripts/Models/base_random_forest.py`
- `Raw RF`
- `PCA RF`
- `RF+DOW`
- `PCA RF+DOW`

`PyScripts/Models/base_SVM.py`
- `Raw Linear SVM`
- `Raw RBF SVM`
- `Raw Poly SVM`

### Model comparison rule
- Within each model script, the final "best model" is chosen by comparing all tuned candidate models produced in that script.
- The shared ranking helper uses weighted average rank on holdout test metrics.
- `test_split_accuracy` has double weight relative to `test_roc_auc_macro`, `test_sensitivity_macro`, and `test_specificity_macro`.
- Any global leaderboard written to `output/*global_model_leaderboard*.csv` is informational only and does not override the plotted/exported winner chosen within each script.

### Estimated runtime for the 8-year dataset
- The estimates below are based on saved run history in `output/8yrs_search_runs.csv` and `output/OLD/results/search_runs.csv`, together with the currently active search grids in `PyScripts/Models/`.
- No rerun was performed to produce these estimates.
- Main tuned pipeline in `PyScripts/Models/All.py` (`SVM.py`, `logistic_regression.py`, `random_forest.py`): approximately 10 to 15 minutes in a typical run, with a wider observed range of about 2 to 14 minutes depending mainly on random-forest search time and available CPU parallelism.
- If running all active model scripts in this repository (`base.py`, `base_SVM.py`, `base_random_forest.py`, `logistic_regression.py`, `SVM.py`, `random_forest.py`), a practical budget is about 20 to 30 minutes.
- Approximate per-script runtime on the 8-year daily dataset:
  - `PyScripts/Models/logistic_regression.py`: about 4 to 10 seconds
  - `PyScripts/Models/SVM.py`: about 10 to 70 seconds
  - `PyScripts/Models/random_forest.py`: about 2 to 12 minutes
  - `PyScripts/Models/base.py`: about 1 to 5 minutes
  - `PyScripts/Models/base_SVM.py`: about 30 to 90 seconds
  - `PyScripts/Models/base_random_forest.py`: about 2 to 10 minutes
