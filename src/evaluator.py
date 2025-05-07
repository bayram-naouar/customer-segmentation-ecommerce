import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_model(df: pd.DataFrame, labels: pd.Series | list, model_name: str) -> dict:
    """
    Evaluate clustering performance using multiple metrics.
    Handles DBSCAN noise points (label -1) appropriately.

    Parameters:
        df (pd.DataFrame): Scaled feature data used for clustering.
        labels (array-like): Cluster labels assigned by the algorithm.

    Returns:
        dict: Dictionary with clustering evaluation scores.
    """
    # DBSCAN may have -1 for noise points, so remove them for the evaluation
    if np.any(labels == -1):
        print("=> [INFO] DBSCAN detected noise points (label = -1).")
        valid_labels = labels[labels != -1]
        valid_data = df[labels != -1]
    else:
        valid_labels = labels
        valid_data = df
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    scores = {
        "Silhouette Score": silhouette_score(valid_data, valid_labels) if n_clusters >= 2 else None,
        "Calinski-Harabasz Index": calinski_harabasz_score(valid_data, valid_labels) if n_clusters >= 2 else None,
        "Davies-Bouldin Index": davies_bouldin_score(valid_data, valid_labels)if n_clusters >= 2 else None
    }

    print(f"=> [INFO] {model_name} Evaluation:")
    for metric, score in scores.items():
        if score:
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: N/A")

    return scores
