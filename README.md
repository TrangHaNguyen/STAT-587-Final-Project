## Project Background

This repository contains an individual version of a project that originally began as a group project. The work was later split into separate individual projects following a mutual decision by the contributors to proceed independently because of differences in objectives, methods, and findings.

The author contributed to the original upstream repository during the shared project stage, but made no further contributions to that repository after the split. From that point onward, this repository has been developed and maintained independently.

Some initial components in this repository originated from the earlier shared project stage, including parts of the data-loading code, model pipeline code, and exploratory data analysis code. The author acknowledges the other contributor’s earlier work on these components. In the present repository, these materials have been revised, extended, or adapted independently.

For reference, the original upstream repository from the earlier shared project stage is available here: [Original Upstream Repository] (https://github.com/Sabuchan123/STAT-587-Final-Project). Following the separation of the project, that repository continued as the other contributor’s independently maintained project, and the shared development stage ended on March 9, 2026.

## Instruction for Replication
(to be added later)
Work flow:
1. Set up the environment to work
Python requirement: use Python 3.11.x for reproducibility (TensorFlow compatibility).

To recreate the Python environment on another computer, run the setup script from the project root:
```bash
PYTHON_BIN=python3.11 bash setup_env.sh
```

This script will:
- create a local `.venv`
- upgrade `pip`
- install all Python dependencies from `requirements.txt`

After setup, activate the environment with:

```bash
source .venv/bin/activate
```

Example entry points:
If you need to compile the report in `LaTex/`, install the LaTeX system packages noted in `requirements.txt`.


2. Downloading data. Code and results are infolder Data
- hourly data: loading_hourly_data.py
- 2 yrs and 8 yrs data: loading_daily_data.py. The output is raw_data_8_years.parquet.
Note: there is slightly differences in the data downloaded for replication on March 10th compared with the original data downloaded by the other contributor in this project.
They can be found in difference.csv and ticker_largest_difference. Potential explanation: different version of yfinance, different options in downloading(less likely as I tried multiple options)

3. Run codes for different part of the report
- EDA part: EDA_simple and EDA_descriptive_stats
4. Run codes for base model of only OHLCV features (and days of week for cyclical adjustment)
