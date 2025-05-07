# main.py

import joblib
import os

from src.data_loader import load_and_clean_data
from src.feature_engineering import compute_rfm, scale_rfm
from src.utils import save_dataframe, ask_yes_no
from src.kmeans_clustering import plot_get_best_k_elbow_method, plot_get_best_k_silhouette_method, compute_kmeans
from src.gmm_clustering import plot_get_best_n_aicbic_method, plot_get_best_n_silhouette_method, compute_gmm
from src.dbscan_clustering import plot_get_best_eps_k_distance_graph_method, plot_get_best_eps_silhouette_method, compute_dbscan, get_eps_range_from_knee
from src.evaluator import evaluate_model
from src.visualization import plot_clusters_2d, plot_clusters_3d

def main():
    # Step 1: Load raw data and clean
    print("[INFO] Loading and cleaning data...")
    df_clean = load_and_clean_data("data/raw/Online Retail.xlsx")
    print("[DONE] Data shape after cleaning:", df_clean.shape, "...")

    # Step 2: Feature engineering (RFM)
    print("[INFO] Generating RFM features...")
    rfm = compute_rfm(df_clean)
    print("[DONE] Data shape after RFM:", rfm.shape, "...")
    print("[INFO] Scaling and saving RFM features...")
    rfm_scaled = scale_rfm(rfm)
    save_dataframe(rfm_scaled, "data/processed/scaled_rfm.csv")
    print("[DONE] RFM features loaded.")

    # Check if user wants to use sa ved models
    kmeans_model_path = "models/kmeans_model.joblib"
    gmm_model_path = "models/gmm_model.joblib"
    dbscan_model_path = "models/dbscan_model.joblib"
    resp = False
    if os.path.exists(kmeans_model_path) and os.path.exists(gmm_model_path) and os.path.exists(dbscan_model_path):
        resp = ask_yes_no("Do you want to use saved models? (y / n) : ")
    if resp:
        kmeans = joblib.load(kmeans_model_path)
        gmm = joblib.load(gmm_model_path)
        dbscan = joblib.load(dbscan_model_path)
        kmeans_labels = kmeans.predict(rfm_scaled.copy())
        gmm_labels = gmm.predict(rfm_scaled.copy())
        dbscan_labels = dbscan.fit_predict(rfm_scaled.copy())
        kmeans_score = evaluate_model(rfm_scaled.copy(), kmeans_labels, "K-Means")
        gmm_score = evaluate_model(rfm_scaled.copy(), gmm_labels, "GMM")
        dbscan_score = evaluate_model(rfm_scaled.copy(), dbscan_labels, "DBSCAN")
    else:
        # Step 3: Train models
        # 3-1: K-MEANS
        print("[INFO] Training K-Means...")
        best_k_elbow = plot_get_best_k_elbow_method(rfm_scaled.copy(), save_path="results/visuals/K-Means Elbow Method.png")
        best_k_silhouette = plot_get_best_k_silhouette_method(rfm_scaled.copy(), save_path="results/visuals/K-Means Silhouette Method.png")
        best_k = max(best_k_elbow, best_k_silhouette)
        print(f"[INFO] Training K-Means with k={best_k}...")
        kmeans, kmeans_labels = compute_kmeans(rfm_scaled.copy(), best_k)
        kmeans_score = evaluate_model(rfm_scaled.copy(), kmeans_labels, "K-Means")
        
        #3-2: GMM
        print("[INFO] Training GMM...")
        best_n_aic, best_n_bic = plot_get_best_n_aicbic_method(rfm_scaled.copy(), save_path="results/visuals/GMM AIC-BIC Method.png")
        best_n_silhouette = plot_get_best_n_silhouette_method(rfm_scaled.copy(), save_path="results/visuals/GMMSilhouette Method.png")
        best_n = max(best_n_aic, best_n_bic, best_n_silhouette)
        print(f"[INFO] Training GMM with n_components={best_n}...")
        gmm, gmm_labels = compute_gmm(rfm_scaled.copy(), best_n)
        gmm_score = evaluate_model(rfm_scaled.copy(), gmm_labels, "GMM")
    
        #3-3: DBSCAN
        print("[INFO] Training DBSCAN...")
        knee_eps, eps_range = get_eps_range_from_knee(rfm_scaled.copy())
        best_eps_silhouette = plot_get_best_eps_silhouette_method(rfm_scaled.copy(), eps_range, save_path="results/visuals/DBSCAN Silhouette Method.png")
        best_eps_knn =plot_get_best_eps_k_distance_graph_method(rfm_scaled.copy(), save_path="results/visuals/DBSCAN Knee Method.png")
        best_eps = (best_eps_silhouette + best_eps_knn) / 2
        print(f"[INFO] Training DBSCAN with eps={best_eps}...")
        dbscan, dbscan_labels = compute_dbscan(rfm_scaled.copy(), best_eps)
        dbscan_score = evaluate_model(rfm_scaled.copy(), dbscan_labels, "DBSCAN")

        # Step 4: Save models
        print("[INFO] Saving models...")
        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/kmeans_model.joblib")
        joblib.dump(gmm, "models/gmm_model.joblib")
        joblib.dump(dbscan, "models/dbscan_model.joblib")
        with open("results/scores.txt", "w") as f:
            f.write(f"K-Means Score: {kmeans_score}\n")
            f.write(f"GMM Score: {gmm_score}\n")
            f.write(f"DBSCAN Score: {dbscan_score}\n")

    # Step 5: Visualize clusters (PCA-2D)
    print("[INFO] Generating cluster 2D visualizations...")
    os.makedirs("results/visuals", exist_ok=True)
    plot_clusters_2d(rfm_scaled.copy(), kmeans_labels, "results/visuals/kmeans_clusters.png", title="K-Means Clusters")
    plot_clusters_2d(rfm_scaled.copy(), gmm_labels, "results/visuals/gmm_clusters.png", title="GMM Clusters")
    plot_clusters_2d(rfm_scaled.copy(), dbscan_labels, "results/visuals/dbscan_clusters.png", title="DBSCAN Clusters")

    # Step 6: Visualize clusters (PCA-3D)
    print("[INFO] Generating cluster 3D visualizations...")
    plot_clusters_3d(rfm_scaled.copy(), kmeans_labels, "results/visuals/kmeans_clusters_3d.png", title="K-Means Clusters")
    plot_clusters_3d(rfm_scaled.copy(), gmm_labels, "results/visuals/gmm_clusters_3d.png", title="GMM Clusters")
    plot_clusters_3d(rfm_scaled.copy(), dbscan_labels, "results/visuals/dbscan_clusters_3d.png", title="DBSCAN Clusters")

    print("[DONE]Pipeline completed successfully!")
if __name__ == "__main__":
    main()
