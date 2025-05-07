# src/kmeans_clustering.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_kmeans(df: pd.DataFrame, k: int, random_state: int = 42) -> tuple:
    """
    Fit KMeans clustering.

    Returns:
        model (KMeans): Fitted KMeans model.
        labels (np.ndarray): Cluster labels.
    """
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(df)
    return model, labels

def plot_get_best_k_elbow_method(df: pd.DataFrame, max_k: int = 10, save_path: str = None) -> int:
    """
    Plot the elbow method to help choose the number of clusters and return the elbow point.
    
    Returns:
        int: The optimal number of clusters (elbow point).
    """
    print("=> [INFO] Plotting Elbow Method...")
    sse = []
    K = range(1, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        sse.append(kmeans.inertia_)

    # Find elbow point: Distance from line method
    # Coordinates of all points
    x = np.array(K)
    y = np.array(sse)

    # Line between first and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Compute distances to the line
    distances = np.abs(np.cross(p2 - p1, p1 - np.vstack((x, y)).T)) / np.linalg.norm(p2 - p1)
    elbow_k = x[np.argmax(distances)]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(K, sse, marker='o')
    plt.axvline(elbow_k, color='red', linestyle='--', label=f"Elbow at k={elbow_k}")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return elbow_k

def plot_get_best_k_silhouette_method(df: pd.DataFrame, k_range=range(2, 11), save_path: str = None) -> int:
    """
    Return best k based on silhouette score.
    """
    print("=> [INFO] Getting best k by silhouette score...")
    best_k = k_range[0]
    best_score = -1
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        score = silhouette_score(df, kmeans.labels_)
        scores.append(score)
        if score > best_score:
            best_k = k
            best_score = score
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, scores, marker='o')
    plt.title("K-Means Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return best_k
