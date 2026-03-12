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
- If you do not want to retrain the models, keep those `.pkl` files and inspect them later using the commented helper examples at the end of [base.py](/Users/tranghanguyen/STAT-587-Final-Project/PyScripts/Models/base.py).
- The model `.pkl` files store fitted estimators; the diagnostics `.pkl` files store the cached bias-variance and train/test curve values used for the figures.


