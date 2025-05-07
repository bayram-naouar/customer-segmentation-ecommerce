# src/dbscan_clustering.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def compute_dbscan(df: pd.DataFrame, eps: float, min_samples: int = 5) -> tuple:
    """
    Fit DBSCAN clustering.

    Returns:
        model (DBSCAN): Fitted DBSCAN model.
        labels (np.ndarray): Cluster labels.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df)
    return model, labels

def get_eps_range_from_knee(df: pd.DataFrame, min_samples: int = 5, num_points: int = 100) -> tuple:
    """
    Estimate a good eps range using the k-distance graph and return the knee point and suggested range.

    Returns:
        knee_eps (float): Estimated best eps from knee point.
        eps_range (list): Range of eps values to try around the knee.
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(df)
    distances, _ = neigh.kneighbors(df)
    k_distances = np.sort(distances[:, -1])

    # Select subset if too large
    if len(k_distances) > num_points:
        k_distances = k_distances[-num_points:]
    
    # Knee point detection using distance from line method
    x = np.arange(len(k_distances))
    y = k_distances
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    distances = np.abs(np.cross(p2 - p1, p1 - np.vstack((x, y)).T)) / np.linalg.norm(p2 - p1)
    knee_index = np.argmax(distances)
    knee_eps = y[knee_index]

    # Define a range around the knee
    eps_range = np.round(np.linspace(knee_eps * 0.7, knee_eps * 1.3, 20), 10)

    return float(knee_eps), eps_range.tolist()


def plot_get_best_eps_silhouette_method(df: pd.DataFrame, eps_range: list, min_samples: int = 5, save_path: str = None) -> float:
    """
    Plot silhouette score for each value in eps_range to choose the best eps.

    Returns:
        float: eps value with highest silhouette score.
    """
    print("=> [INFO] Getting best eps by silhouette score...")
    best_eps = eps_range[0]
    best_score = -1
    scores = []

    for eps in eps_range:
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
        labels = model.labels_
        # Ignore cases where DBSCAN assigns all as noise or one cluster
        if len(set(labels)) <= 1 or (set(labels) == {-1}):
            scores.append(-1)
            continue
        score = silhouette_score(df, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_eps = eps

    plt.figure(figsize=(8, 4))
    plt.plot(eps_range, scores, marker='o')
    plt.title(" DBSCAN Silhouette Score vs Epsilon (eps)")
    plt.xlabel("Epsilon (eps)")
    plt.ylabel("Silhouette Score")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return best_eps

def plot_get_best_eps_k_distance_graph_method(df: pd.DataFrame, k: int = 5, save_path: str = None) -> float:
    """
    Plot the k-distance graph to help estimate a good eps, and return the recommended eps.

    Args:
        df (pd.DataFrame): Scaled input data.
        k (int): Number of neighbors (usually set to min_samples).
        save_path (str, optional): If given, saves the plot.

    Returns:
        float: Recommended eps value based on elbow detection.
    """
    print("=> [INFO] Plotting k-distance graph...")
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(df)
    distances, _ = neigh.kneighbors(df)
    k_distances = np.sort(distances[:, -1])

    # Elbow detection via max curvature method
    n_points = len(k_distances)
    all_coords = np.vstack((np.arange(n_points), k_distances)).T
    first = all_coords[0]
    last = all_coords[-1]
    line_vec = last - first
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = all_coords - first
    scalar_product = np.dot(vec_from_first, line_vec_norm)
    proj = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - proj
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    elbow_idx = np.argmax(dist_to_line)
    recommended_eps = k_distances[elbow_idx]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances, label='k-distances')
    plt.axvline(x=elbow_idx, color='r', linestyle='--', label=f"Elbow at idx {elbow_idx}")
    plt.axhline(y=recommended_eps, color='g', linestyle='--', label=f"eps ≈ {recommended_eps:.4f}")
    plt.title(f"DBSCAN {k}-Distance Graph")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return recommended_eps

