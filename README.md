## Project Background

This repository contains an individual version of a project that originally began as a group project. The work was later split into separate individual projects following a mutual decision by the contributors to proceed independently because of differences in objectives, methods, and findings.

The author contributed to the original upstream repository during the shared project stage, but made no further contributions to that repository after the split. From that point onward, this repository has been developed and maintained independently.

Some initial components in this repository originated from the earlier shared project stage, including parts of the data-loading code, model pipeline code, and exploratory data analysis code. The author acknowledges the other contributor’s earlier work on these components. In the present repository, these materials have been revised, extended, or adapted independently.

For reference, the original upstream repository from the earlier shared project stage is available here: [Original Upstream Repository] (https://github.com/Sabuchan123/STAT-587-Final-Project). Following the separation of the project, that repository continued as the other contributor’s independently maintained project, and the shared development stage ended on March 9, 2026.

## Instruction for Replication
(to be added later)
Work flow:
### 1. Set up the environment to work

> ⚠️ **Two separate environments are required.** TensorFlow 2.17.x (used by
> the neural network scripts) requires `numpy < 2.0`, which conflicts with the
> `numpy==2.4.2` needed by all other models. Installing both into the same
> virtual environment will cause dependency conflicts. Always use:
>
> | Environment | Scripts | Python | Requirements file |
> |---|---|---|---|
> | `.venv_nn` | `NN.py`, `base_NN.py` | 3.11 (via uv) | `requirements_nn.txt` |
> | `.venv` | everything else | 3.12 | `requirements.txt` |

#### On macOS — Neural Network models (Python 3.11 + TensorFlow)
```bash
brew install python@3.11
PYTHON_BIN=python3.11 bash setup_env.sh   # creates .venv with Python 3.11
source .venv/bin/activate
pip install -r requirements_nn.txt
```

#### On macOS — All other models (Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### On ORCA — see `ORCA.txt` sections 2A and 2B for cluster-specific setup.

If you need to compile the report in `LaTex/`, install the LaTeX system packages noted at the bottom of `requirements.txt`.

#### Example after setting up environment
```bash
source .venv_nn/bin/activate        # for NN scripts
python PyScripts/Models/NN.py

source .venv/bin/activate           # for all other scripts
python PyScripts/Data/loading_daily_data.py
```

### 2. Downloading data. Code and results are infolder Data
- hourly data: loading_hourly_data.py
- 2 yrs and 8 yrs data: loading_daily_data.py. The output is raw_data_8_years.parquet.
Note: there is slightly differences in the data downloaded for replication on March 10th compared with the original data downloaded by the other contributor in this project.
They can be found in difference.csv and ticker_largest_difference. Potential explanation: different version of yfinance, different options in downloading(less likely as I tried multiple options)

### 3. Run codes for EDA section
- EDA part: EDA_simple



### 4. Run codes for different models

> **Checkpoints are not included in this repository** (nearly 1 GB — too large
> for GitHub). All runtimes below assume a **fresh run with no cached checkpoints**.
> On a rerun where checkpoints exist locally, most stages complete in under
> 5 minutes. On ORCA, use `run_allothers.slurm` (other models) and
> `run_nn.slurm` (NN) — both support `--resume` to skip already-completed stages.


- Run the code of different models in the order:

base.py → base_random_forest.py → base_SVM.py → base_NN.py

Estimated runtime on the 8-year dataset (fresh run, no checkpoints, measured on ORCA):

| Script | Fresh run | With checkpoints |
|---|---|---|
| `base.py` | ~6 min | ~1 min |
| `base_random_forest.py` | ~48 min | ~1 min |
| `base_SVM.py` | ~7 min | <1 min |
| `base_NN.py` | epoch CV sweep ~2 min + model eval† | <1 min |

† Measured on local Mac (2026-03-15). Epoch CV sweep covers 5 values × 5 folds × 3 architectures. Full fresh-run training time was not logged; ORCA runtimes may differ. Requires the separate `.venv_nn` environment (Python 3.11 + TensorFlow 2.17.x).

### 5. Run code for engineered data
logistic_regression.py → random_forest.py → SVM.py → NN.py

Estimated runtime on the 8-year dataset (fresh run, no checkpoints, measured on ORCA):

| Script | Fresh run | With checkpoints |
|---|---|---|
| `logistic_regression.py` | ~9 min | ~1 min |
| `random_forest.py` | **9+ hrs** | ~5 min |
| `SVM.py` | ~56 min | <1 min |
| `NN.py` | epoch CV sweep ~12 min + model eval† | <1 min |

† Measured on local Mac (2026-03-15). Epoch CV sweep covers 5 values × 5 folds × 3 architectures. Full fresh-run training time was not logged; ORCA runtimes may differ. Requires the separate `.venv_nn` environment (Python 3.11 + TensorFlow 2.17.x).

**Total fresh-run estimate on ORCA:**
- Other models (`run_allothers.slurm`): ~11 hrs (engineered RF dominates; the 18 hr time limit is appropriate)
- NN models (`run_nn.slurm`): epoch CV sweeps ~14 min total (local Mac); full training time not logged — 8 hr time limit set conservatively

### 6. Constructing vizualization plot for prediction
predictionvizualization
