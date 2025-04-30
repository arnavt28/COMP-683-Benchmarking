import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FlowCal
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import pickle

# ----------------- CONFIG -----------------
FILENAMES_TRANSFORM = [
    "FlowRepository_FR-FCM-ZZPH_files/Levine_13dim.fcs",
    "FlowRepository_FR-FCM-ZZPH_files/Levine_32dim.fcs",
    "FlowRepository_FR-FCM-ZZPH_files/Mosmann_rare.fcs",
    "FlowRepository_FR-FCM-ZZPH_files/Nilsson_rare.fcs",
    "FlowRepository_FR-FCM-ZZPH_files/Samusik_01.fcs",
    "FlowRepository_FR-FCM-ZZPH_files/Samusik_all.fcs"
]

OUTPUT_DIR = "benchmark_results_dbscan_auto_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- HELPER FUNCTIONS -----------------
def match_clusters(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred])

def calculate_purity(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def estimate_eps(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])
    return np.percentile(k_distances, 90)  # 90th percentile as eps estimate

# ----------------- MAIN SCRIPT -----------------
summary_rows = []

for input_file in tqdm(FILENAMES_TRANSFORM, desc="Running DBSCAN"):
    dataset_name = os.path.basename(input_file).replace('.fcs', '')
    print(f"\n📂 Processing {dataset_name}")

    try:
        s = FlowCal.io.FCSData(input_file)
        data_array = np.asarray(s)
    except Exception as e:
        print(f"❌ Failed to load {input_file}: {e}")
        continue

    if "Levine_32dim" in dataset_name:
        X = data_array[:, :-2]
        y_true = data_array[:, -2]
    else:
        X = data_array[:, :-1]
        y_true = data_array[:, -1]

    # Remove NaNs
    valid_rows = ~np.isnan(X).any(axis=1)
    X = X[valid_rows]
    y_true = y_true[valid_rows]

    # Estimate parameters
    n_features = X.shape[1]
    min_samples = 2 * n_features
    eps = estimate_eps(X, k=2 * n_features - 1)

    # Run DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    labeled_indices = ~np.isnan(y_true)
    y_true_clean = y_true[labeled_indices].astype(int)
    y_pred_clean = labels[labeled_indices]

    # Compute metrics
    nmi = normalized_mutual_info_score(y_true_clean, y_pred_clean)
    ari = adjusted_rand_score(y_true_clean, y_pred_clean)
    purity = calculate_purity(y_true_clean, y_pred_clean)
    f1 = f1_score(y_true_clean, match_clusters(y_true_clean, y_pred_clean), average='weighted')
    conf_mat = confusion_matrix(y_true_clean, match_clusters(y_true_clean, y_pred_clean))

    results = {
        "dataset": dataset_name,
        "method": "DBSCAN",
        "eps": eps,
        "min_samples": min_samples,
        "nmi": nmi,
        "ari": ari,
        "purity": purity,
        "f1_score": f1
    }

    # Save results
    output_prefix = f"{dataset_name}_DBSCAN_eps{eps:.2f}_min{min_samples}"
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, f"{output_prefix}.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, f"{output_prefix}_full.pkl"), "wb") as f:
        pickle.dump({
            "params": {"eps": eps, "min_samples": min_samples},
            "confusion_matrix": conf_mat,
            "metrics": results
        }, f)

    summary_rows.append(results)

# Save combined CSV
combined_df = pd.DataFrame(summary_rows)
combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_dbscan_summary.csv"), index=False)
print("\n✅ All datasets processed. Results saved to:", OUTPUT_DIR)
