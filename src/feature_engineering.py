# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import RobustScaler

def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features for each customer.

    Parameters:
        df (pd.DataFrame): Cleaned retail transaction data.
        reference_date (pd.Timestamp, optional): Date to compute recency from.
            Defaults to max InvoiceDate in the dataset + 1 day.

    Returns:
        pd.DataFrame: DataFrame with RFM features indexed by CustomerID.
    """
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Compute RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'UnitPrice': lambda x: (df.loc[x.index, 'Quantity'] * x).sum()
    })

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'UnitPrice': 'Monetary'
    }, inplace=True)

    return rfm

def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply RobustScaler to RFM features.

    Parameters:
        rfm (pd.DataFrame): DataFrame with RFM features.

    Returns:
        pd.DataFrame: Scaled RFM features with same index.
    """
    scaler = RobustScaler()
    rfm_scaled = pd.DataFrame(scaler.fit_transform(rfm), 
                              index=rfm.index, 
                              columns=rfm.columns)
    return rfm_scaled