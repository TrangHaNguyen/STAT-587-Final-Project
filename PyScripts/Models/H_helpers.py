from pathlib import Path
import os
import csv
import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

def get_cwd(base_folder, max_lookback: int =5):
    cwd=Path.cwd()
    for _ in range(max_lookback): 
        if cwd.name!=base_folder:
            cwd=cwd.parent
        else:
            break
    else:
        raise FileNotFoundError("Could not find correct workspace folder.")
    return cwd

def log_result(result_dict: dict, directory: Path, file_name: str):
    file_exists=os.path.isfile(directory / file_name)
    with open(directory / file_name, 'a', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=result_dict.keys())
        if (not file_exists):
            writer.writeheader()
        writer.writerow(result_dict)
        print("Wrote to", directory / file_name)

def get_model_params(model: BaseEstimator):
    params=model.get_params()
    name=model.__class__.__name__
    print(name)
    if (name=='Pipeline'):
        model=model.named_steps['classifier']
        params=model.get_params()
        name=model.__class__.__name__
    if (name=='SVC'):
        return f"C={params.get('C')}, kernel={params.get('kernel')}, gamma={params.get('gamma')}"
    elif (name=='RandomForestClassifier'):
        return f"n_est={params.get('n_estimators')}, depth={params.get('max_depth')}, feat={params.get('max_features')}"
    elif (name=='LogisticRegression'):
        return f"C={params.get('C')}, l1_ratio={params.get('l1_ratio')}"
    else:
        return str(params)
    
def get_model_params_grid(grid: GridSearchCV):
    params=grid.best_params_
    return_string=""
    for param in params:
        return_string += f"{param}={params[param]}"
    return return_string

def append_params_to_dict(dict_: dict, model: BaseEstimator):
    dict_['parameters']=get_model_params(model)
    return dict_

def append_grid_params_to_dict(dict_: dict, grid: GridSearchCV):
    dict_['parameters']=get_model_params_grid(grid)
    return dict_