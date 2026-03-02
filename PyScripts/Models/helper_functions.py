from pathlib import Path

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