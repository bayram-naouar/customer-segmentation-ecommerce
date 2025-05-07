# src/data_loader.py

import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the Online Retail dataset from an Excel file and perform basic cleaning.

    Parameters:
        filepath (str): Path to the Excel (.xlsx) file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Load the Excel file
    df = pd.read_excel(filepath)

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove negative or zero quantities
    df = df[df['Quantity'] > 0]

    # Remove negative or zero unit prices
    df = df[df['UnitPrice'] > 0]

    # Convert CustomerID to string to treat it as categorical
    df['CustomerID'] = df['CustomerID'].astype(str)

    return df.copy()
