import numpy as np
import pandas as pd
import time
import FlowCal
import sys
import os
import subprocess
import psutil  # New for memory measurement

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment


def match_clusters(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_matched = np.array([mapping.get(label, label) for label in y_pred])
    return y_pred_matched

def calculate_purity(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def benchmark_clustering(X, y_true_full, clustering_func, method_name, n_clusters=None):
    results = {}

    # Measure memory before clustering
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    start_time = time.time()
    if n_clusters is not None:
        y_pred = clustering_func(X, n_clusters)
    else:
        y_pred = clustering_func(X)
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

def kmeans_clustering(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(X)

def dbscan_clustering(X):
    model = DBSCAN(eps=2, min_samples=5)
    return model.fit_predict(X)

def hierarchical_clustering(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)


def tsne_kmeans_clustering(X, n_clusters):
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(X_embedded)

clustering_methods = {
    'KMeans': (kmeans_clustering, True),
    'DBSCAN': (dbscan_clustering, False),
    'Hierarchical': (hierarchical_clustering, True),
    'tSNE+KMeans': (tsne_kmeans_clustering, True)
}

# Node info gathering
def get_node_info():
    try:
        cpu_info = subprocess.check_output('lscpu', shell=True).decode()
        mem_info = subprocess.check_output('free -h', shell=True).decode()
        return {"cpu_info": cpu_info, "memory_info": mem_info}
    except Exception as e:
        return {"cpu_info": "Unavailable", "memory_info": "Unavailable"}


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    try:
        s = FlowCal.io.FCSData(input_file)
        data_array = np.asarray(s)

        dataset_name = os.path.basename(input_file).replace('.fcs', '')

        # Handle special case for Levine_32dim files
        if "Levine_32dim" in dataset_name:
            X_full = data_array[:, :-2]  # All marker channels except last two
            y_true_full = data_array[:, -2]  # Second to last column is labels
        else:
            X_full = data_array[:, :-1]
            y_true_full = data_array[:, -1]

        # Filter out rows with NaN in X
        valid_rows = ~np.isnan(X_full).any(axis=1)
        X = X_full[valid_rows]
        y_true_full = y_true_full[valid_rows]

    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    labeled_indices = ~np.isnan(y_true_full)
    n_clusters = len(np.unique(y_true_full[labeled_indices].astype(int)))

    results = {}

    for method_name, (method_func, needs_n_clusters) in clustering_methods.items():
        try:
            if needs_n_clusters:
                metrics = benchmark_clustering(X, y_true_full, method_func, method_name, n_clusters=n_clusters)
            else:
                metrics = benchmark_clustering(X, y_true_full, method_func, method_name)
            results[method_name] = metrics
        except Exception as e:
            print(f"  {method_name} failed on {dataset_name}: {e}")
            results[method_name] = None

    node_specs = get_node_info()
    full_results = {
        "node_specs": node_specs,
        "benchmark_results": results
    }

    output_full_pickle = os.path.join(output_dir, f"{dataset_name}_benchmark_full.pkl")
    output_csv = os.path.join(output_dir, f"{dataset_name}_benchmark.csv")

    import pickle

    with open(output_full_pickle, "wb") as f:
        pickle.dump(full_results, f, protocol=4)

    flattened = pd.DataFrame([
        {"dataset": dataset_name, "method": method, **metrics}
        for method, metrics in results.items() if metrics is not None
    ])
    flattened.to_csv(output_csv, index=False)

    print(f"Finished benchmarking {dataset_name}. Saved full pkl and CSV.")