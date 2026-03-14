from pathlib import Path
import os

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


def log_result(result_dict: dict, directory: Path, file_name: str) -> None:
    output_path = directory / file_name
    file_exists = os.path.isfile(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    export_key_map = {
        "train_avg_accuracy": "cv_train_plain_accuracy_mean",
        "train_std_accuracy": "cv_train_plain_accuracy_std",
        "validation_avg_accuracy": "cv_validation_plain_accuracy_mean",
        "validation_std_accuracy": "cv_validation_plain_accuracy_std",
        "cv_test_sd_error": "cv_validation_plain_accuracy_sd_reported",
        "test_split_accuracy": "holdout_test_plain_accuracy",
        "test_misclassification_error": "holdout_test_plain_misclassification_error",
        "mwfv_train_avg_accuracy": "rolling_train_plain_accuracy_mean",
        "mwfv_train_std_accuracy": "rolling_train_plain_accuracy_std",
        "mwfv_test_avg_accuracy": "rolling_test_plain_accuracy_mean",
        "mwfv_test_std_accuracy": "rolling_test_plain_accuracy_std",
    }
    formatted_dict = {}
    for key, value in result_dict.items():
        export_key = export_key_map.get(key, key)
        if isinstance(value, float):
            formatted_dict[export_key] = round(value, 3)
        else:
            formatted_dict[export_key] = value
    new_row = pd.DataFrame([formatted_dict])
    if file_exists:
        existing = pd.read_csv(output_path, engine="python", on_bad_lines="skip")
        existing = existing.rename(columns=export_key_map)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
        combined.to_csv(output_path, index=False)
    else:
        new_row.to_csv(output_path, index=False)
    print("Downloaded results to", output_path)


def get_model_params(model: BaseEstimator) -> str:
    params = model.get_params()
    name = model.__class__.__name__
    if name == 'Pipeline':
        model = model.named_steps['classifier']
        params = model.get_params()
        name = model.__class__.__name__
    if name == 'SVC':
        return f"C={params.get('C')}, kernel={params.get('kernel')}, gamma={params.get('gamma')}"
    if name == 'RandomForestClassifier':
        return f"n_est={params.get('n_estimators')}, depth={params.get('max_depth')}, feat={params.get('max_features')}"
    if name == 'LogisticRegression':
        return f"C={params.get('C')}, l1_ratio={params.get('l1_ratio')}"
    return str(params)


def get_model_params_grid(grid: GridSearchCV) -> str:
    params = grid.best_params_
    return_string = ""
    for param in params:
        return_string += f"{param}={params[param]}"
    return return_string


def append_params_to_dict(dict_: dict, model: BaseEstimator) -> dict:
    dict_['parameters'] = get_model_params(model)
    return dict_


def append_grid_params_to_dict(dict_: dict, grid: GridSearchCV) -> dict:
    dict_['parameters'] = get_model_params_grid(grid)
    return dict_


# Legacy-only helper functions moved out of H_eval.py.
# Original location: PyScripts/Models/H_eval.py
def display_feat_importances_tree(model, X: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    model_feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    model_feature_df.head(n=n).plot(kind='barh', x="Feature", y="Importance")
    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_feature_df


# Legacy-only helper functions moved out of H_eval.py.
# Original location: PyScripts/Models/H_eval.py
def display_coef_importances_regression(model, X: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    importances = pd.Series(model.coef_[0]).abs().to_numpy()
    model_coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coef': importances
    }).sort_values(by='Coef', ascending=False)
    model_coef_df.head(n=n).plot(kind='barh', x="Feature", y="Coef")
    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_coef_df
