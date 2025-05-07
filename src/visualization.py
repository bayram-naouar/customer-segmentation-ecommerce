# src/visualization.py

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D     
import matplotlib.pyplot as plt

def plot_clusters_2d(df: pd.DataFrame, labels, save_path: str = None, title: str = "Cluster Visualization"):
    """
    Perform PCA on the provided dataframe, plot the 2D clusters with hue based on labels, 
    and optionally save the output.

    Parameters:
        df (pd.DataFrame): Scaled feature dataframe (a copy is recommended).
        labels (array-like): Cluster labels.
        save_path (str, optional): If provided, save the output plot to this path.
        title (str): Title of the plot.
    """
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(df)
    
    # Create a DataFrame with PCA components and cluster labels
    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    # Using seaborn for a nicer scatter plot with hue
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=plot_df, palette="viridis", s=60)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()

def plot_clusters_3d(data: pd.DataFrame, labels: np.ndarray, save_path: str = None, title: str = "3D Cluster Plot"):
    """
    Reduce data to 3D using PCA and plot clusters.
    
    Args:
        data (pd.DataFrame): Scaled data.
        labels (np.ndarray): Cluster labels.
        save_path (str): Optional. If provided, saves the plot to this path.
        title (str): Title of the plot.
    """
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.7
    )

    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
