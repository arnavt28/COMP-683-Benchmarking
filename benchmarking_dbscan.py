# benchmark_dbscan_only.py

import numpy as np
import pandas as pd
import time
import FlowCal
import sys
import os
import subprocess
import psutil  # For memory measurement
import pickle
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# --- Helper functions ---

def match_clusters(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_matched = np.array([mapping.get(label, label) for label in y_pred])
    return y_pred_matched

def calculate_purity(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def benchmark_dbscan(X, y_true_full, eps, min_samples):
    results = {}
    
    # Measure memory before clustering
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    start_time = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X)
    end_time = time.time()

    # Measure memory after clustering
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    peak_memory_usage = mem_after - mem_before

    labeled_indices = ~np.isnan(y_true_full)
    y_true = y_true_full[labeled_indices].astype(int)
    y_pred_labeled = y_pred[labeled_indices]

    runtime = end_time - start_time
    nmi = normalized_mutual_info_score(y_true, y_pred_labeled)
    ari = adjusted_rand_score(y_true, y_pred_labeled)
    purity = calculate_purity(y_true, y_pred_labeled)

    y_pred_matched = match_clusters(y_true, y_pred_labeled)
    f1 = f1_score(y_true, y_pred_matched, average='weighted')

    conf_mat = confusion_matrix(y_true, y_pred_matched)

    results['runtime'] = runtime
    results['memory_usage_MB'] = peak_memory_usage
    results['nmi'] = nmi
    results['ari'] = ari
    results['purity'] = purity
    results['f1_score'] = f1
    results['confusion_matrix'] = conf_mat

    return results

def get_node_info():
    try:
        cpu_info = subprocess.check_output('lscpu', shell=True).decode()
        mem_info = subprocess.check_output('free -h', shell=True).decode()
        return {"cpu_info": cpu_info, "memory_info": mem_info}
    except Exception as e:
        return {"cpu_info": "Unavailable", "memory_info": "Unavailable"}

# --- Main ---

if __name__ == "__main__":
    # Expect 4 command-line arguments:
    # 1. input_file (.fcs)
    # 2. output_dir
    # 3. eps value
    # 4. min_samples value

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    eps = float(sys.argv[3])
    min_samples = int(sys.argv[4])

    os.makedirs(output_dir, exist_ok=True)

    try:
        s = FlowCal.io.FCSData(input_file)
        data_array = np.asarray(s)

        dataset_name = os.path.basename(input_file).replace('.fcs', '')

        if "Levine_32dim" in dataset_name:
            X_full = data_array[:, :-2]  # Special case
            y_true_full = data_array[:, -2]
        else:
            X_full = data_array[:, :-1]
            y_true_full = data_array[:, -1]

        # Remove rows with NaN
        valid_rows = ~np.isnan(X_full).any(axis=1)
        X = X_full[valid_rows]
        y_true_full = y_true_full[valid_rows]

    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Run DBSCAN with specified parameters
    try:
        metrics = benchmark_dbscan(X, y_true_full, eps=eps, min_samples=min_samples)
    except Exception as e:
        print(f"DBSCAN failed on {dataset_name} with eps={eps}, min_samples={min_samples}: {e}")
        sys.exit(1)

    # Save results
    node_specs = get_node_info()

    results = {
        "node_specs": node_specs,
        "dbscan_params": {"eps": eps, "min_samples": min_samples},
        "benchmark_results": metrics
    }

    output_prefix = f"{dataset_name}_DBSCAN_eps{eps}_min{min_samples}"

    with open(os.path.join(output_dir, f"{output_prefix}_full.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=4)

    flattened = pd.DataFrame([{**{"dataset": dataset_name, "method": "DBSCAN"}, **metrics, "eps": eps, "min_samples": min_samples}])
    flattened.to_csv(os.path.join(output_dir, f"{output_prefix}.csv"), index=False)

    print(f"✅ Finished DBSCAN benchmarking {dataset_name} with eps={eps}, min_samples={min_samples}.")
