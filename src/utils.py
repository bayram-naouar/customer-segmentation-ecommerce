# src/utils.py

import pandas as pd

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        path (str): File path to save to.
    """
    df.to_csv(path, index=True)

def ask_yes_no(prompt: str, default: str = None) -> bool:
    """
    Prompt the user for a yes/no answer safely.

    Parameters:
        prompt (str): The message displayed to the user.
        default (str, optional): Default choice if user hits Enter ('y' or 'n').

    Returns:
        bool: True if user answers yes, False otherwise.
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    if default:
        default = default.lower()
        if default not in valid:
            raise ValueError("Default must be 'y' or 'n'")
        prompt_suffix = " [Y/n] " if default == 'y' else " [y/N] "
    else:
        prompt_suffix = " [y/n] "

    while True:
        choice = input(prompt + prompt_suffix).strip().lower()
        if not choice and default:
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'y' or 'n' (or 'yes' or 'no').")