# 🛍️ Customer Segmentation with Clustering Models

This project performs customer segmentation using unsupervised learning techniques on the UK-based Online Retail dataset. The goal is to identify customer clusters based on RFM (Recency, Frequency, Monetary) features to support marketing strategy, personalization, and business insights.

## 📊 Clustering Models Used

- **K-Means Clustering**
- **Gaussian Mixture Models (GMM)**
- **DBSCAN (Density-Based Spatial Clustering)**

Each model is tuned using multiple strategies (e.g., Elbow Method, Silhouette Score, AIC/BIC, K-Distance Knee) and compared using clustering performance metrics.

---

## 🧠 Workflow Overview

1. **Data Loading and Cleaning**
   - Load raw data from `data/raw/Online Retail.xlsx`
   - Clean and preprocess the data (remove nulls, invalid entries)

2. **Feature Engineering**
   - Calculate RFM features per customer
   - Standardize RFM features for clustering

3. **Model Training / Loading**
   - Ask whether to load saved models or retrain
   - K-Means: best k from elbow and silhouette
   - GMM: best n from AIC/BIC and silhouette
   - DBSCAN: best ε from silhouette and knee distance graph

4. **Model Evaluation**
   - Evaluate clusters using silhouette score and other internal metrics
   - Save models to `models/` and metrics to `results/scores.txt`

5. **Visualization**
   - Visualize clusters in 2D and 3D using PCA
   - Save plots under `results/visuals/`

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw/                      # Raw Excel file goes here
│   └── processed/               # Scaled RFM CSV will be saved here
├── models/                      # Trained clustering models (.joblib)
├── results/
│   ├── visuals/                 # All plots (elbow, silhouette, clusters)
│   └── scores.txt               # Model evaluation scores
├── notebooks/                   # (Optional) Exploratory notebooks
├── src/                         # Core codebase
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── kmeans_clustering.py
│   ├── gmm_clustering.py
│   ├── dbscan_clustering.py
│   ├── evaluator.py
│   ├── utils.py
│   └── visualization.py
├── main.py                      # Pipeline entrypoint
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Add Dataset

Download the **Online Retail Dataset** from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) and place it in:

```
data/raw/Online Retail.xlsx
```

### 3. Run the Pipeline

```bash
python main.py
```

### 4. View Results

- 📈 Model performance in `results/scores.txt`
- 📊 Cluster plots in `results/visuals/`
- 💾 Trained models in `models/`

---

## 📌 Notes

- If saved models already exist, you'll be prompted to reuse them or retrain from scratch.
- DBSCAN’s best ε is the **average** of silhouette-based and k-distance-based values for more stable clustering.
- All visualizations are saved automatically; no GUI or manual plotting required.

---

## 🧩 Future Work

- Integrate hyperparameter tuning automation
- Add t-SNE visualizations
- Export cluster labels with customer IDs for business use
- Add comparison with supervised labeling (if any labels are added)

---

## 📬 Contact

Project by **[Your Name]**  
For questions, feel free to open an [Issue](https://github.com/your-repo/issues) or reach out via GitHub.
