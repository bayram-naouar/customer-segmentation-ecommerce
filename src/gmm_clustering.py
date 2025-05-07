# src/gmm_clustering.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def compute_gmm(df: pd.DataFrame, n_components: int, random_state: int = 42) -> tuple:
    """
    Fit Gaussian Mixture Model.

    Returns:
        model (GaussianMixture): Fitted GMM model.
        labels (np.ndarray): Cluster labels.
    """
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(df)
    return model, labels

def plot_get_best_n_aicbic_method(df: pd.DataFrame, max_components: int = 10, save_path: str = None) -> tuple:
    """
    Plot AIC and BIC to help choose the number of GMM components.

    Returns:
        tuple: (best_aic_n, best_bic_n)
    """
    print("=> [INFO] Plotting AIC and BIC...")
    aic_scores = []
    bic_scores = []
    n_values = range(1, max_components + 1)

    for n in n_values:
        gmm = GaussianMixture(n_components=n, random_state=42).fit(df)
        aic_scores.append(gmm.aic(df))
        bic_scores.append(gmm.bic(df))

    best_aic_n = n_values[np.argmin(aic_scores)]
    best_bic_n = n_values[np.argmin(bic_scores)]

    plt.figure(figsize=(8, 4))
    plt.plot(n_values, aic_scores, marker='o', label="AIC")
    plt.plot(n_values, bic_scores, marker='o', label="BIC")
    plt.axvline(best_aic_n, linestyle='--', color='green', label=f"Best AIC: {best_aic_n}")
    plt.axvline(best_bic_n, linestyle='--', color='red', label=f"Best BIC: {best_bic_n}")
    plt.title("AIC and BIC vs Number of GMM Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return best_aic_n, best_bic_n

def plot_get_best_n_silhouette_method(df: pd.DataFrame, n_range=range(2, 11), save_path: str = None) -> int:
    """
    Return best n_components for GMM based on silhouette score.

    Returns:
        int: The best number of components.
    """
    print("=> [INFO] Getting best number of components by silhouette score...")
    best_n = n_range[0]
    best_score = -1
    scores = []

    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42).fit(df)
        labels = gmm.predict(df)
        score = silhouette_score(df, labels)
        scores.append(score)
        if score > best_score:
            best_n = n
            best_score = score

    plt.figure(figsize=(8, 4))
    plt.plot(n_range, scores, marker='o')
    plt.title("GMMSilhouette Score")
    plt.xlabel("Number of Components")
    plt.ylabel("Silhouette Score")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return best_n
