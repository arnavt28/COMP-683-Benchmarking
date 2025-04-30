import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

# Thank you to 53RT's repo at https://github.com/53RT/Highly-Connected-Subgraphs-Clustering-HCS
# for giving me the idea to use networkx's functionality to interact with Graph objects.

def split_into_connected_subgraphs(G: nx.Graph) -> list[nx.Graph]:
    S = []
    for c in nx.connected_components(G):
        subgraph = nx.Graph()
        subgraph.add_nodes_from(c)
        subgraph.add_edges_from((n, nbr, d) for n, nbr, d in G.edges(c, data=True) if nbr in c)
        S.append(subgraph)
    return S

def hcs(G: nx.Graph, convergence_threshold: int) -> nx.Graph:
    if G.number_of_nodes() < 2:
        return G
    assert nx.is_connected(G)
    _, partition = nx.stoer_wagner(G)
    cutset = [(u, v) for u, v in G.edges() if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0])]
    if len(list(cutset)) > (G.number_of_nodes() // convergence_threshold):
        return G
    else:
        G.remove_edges_from(cutset)
        connected_subgraphs = split_into_connected_subgraphs(G)
        clustered_subgraphs = [None] * len(connected_subgraphs)
        index = 0
        for subgraph in connected_subgraphs:
            clustered_subgraph = hcs(subgraph, convergence_threshold)
            clustered_subgraphs[index] = clustered_subgraph
            index += 1

        return nx.union_all(clustered_subgraphs)

def improved_hcs(connected_subgraphs: list[nx.Graph], full_graph: nx.Graph, convergence_threshold: int) -> nx.Graph:
    clustered_subgraphs = []
    for connected_subgraph in connected_subgraphs:
        hcs_result = hcs(connected_subgraph, convergence_threshold)
        hcs_result_split = split_into_connected_subgraphs(hcs_result)
        singletons = [cluster for cluster in hcs_result_split if cluster.number_of_nodes() == 1]
        not_singleton_clusters = [cluster for cluster in hcs_result_split if cluster.number_of_nodes() != 1]
        final_clusters = not_singleton_clusters  
        for singleton in singletons:
            for singleton_node in singleton.nodes():
                neighbor_count = [0] * (len(not_singleton_clusters) + 1)
                for (u, v, w) in full_graph.edges(data=True):
                    if u == singleton_node:
                        for i in range(len(not_singleton_clusters)):
                            if v in not_singleton_clusters[i].nodes():
                                neighbor_count[i] += 1
                        for i in range(len(singletons)):
                            if v in singletons[i].nodes():
                                neighbor_count[len(neighbor_count) - 1] += 1
                    elif v == singleton_node:
                        for i in range(len(not_singleton_clusters)):
                            if u in not_singleton_clusters[i].nodes():
                                neighbor_count[i] += 1
                        for i in range(len(singletons)):
                            if u in singletons[i].nodes():
                                neighbor_count[len(neighbor_count) - 1] += 1
                max_neighbors = max(neighbor_count)
                destination_index = neighbor_count.index(max_neighbors)
                if destination_index < (len(neighbor_count) - 1):
                    not_singleton_clusters[destination_index].add_node(singleton_node)
                else:
                    final_clusters.append(singleton)
        clustered_subgraphs.append(final_clusters)
    return clustered_subgraphs
        
def hcs_full(feature_matrix: np.ndarray, num_neighbors: int = 5, min_edge_weight: float = 0.3, convergence_threshold: int = 2, mode: str = "default") -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='kd_tree', metric='minkowski').fit(feature_matrix)
    distances, indices = nbrs.kneighbors(feature_matrix)
    edges = []
    for i in range(len(indices)):
        for j in range(1, len(indices[i])):
            edges.append(f"{indices[i][0]} {indices[i][j]} {1 / distances[i][j]}")

    G: nx.Graph = nx.parse_edgelist(edges, nodetype=int, data=(("weight", float),))

    if min_edge_weight != 0:
        edges_to_delete = []
        for (u, v, w) in G.edges(data=True):
            if w['weight'] < min_edge_weight:
                edges_to_delete.append((u, v))
        G.remove_edges_from(edges_to_delete)

    assert G.number_of_nodes() == len(feature_matrix)
    
    connected_subgraphs = split_into_connected_subgraphs(G)
    clustered_subgraphs = []
    if mode == "default":
        for subgraph in connected_subgraphs:
            clustered_subgraphs.append(hcs(subgraph, convergence_threshold))
    elif mode == "improved":
        clustered_subgraphs = improved_hcs(connected_subgraphs, G, convergence_threshold)
    elif mode == "none": # Baseline kNN
        clustered_subgraphs = connected_subgraphs
    if mode != "improved":
        labels = np.zeros(len(feature_matrix))
        label = 1
        for clustered_subgraph in clustered_subgraphs:
            clustered_subgraph_split = split_into_connected_subgraphs(clustered_subgraph)
            for cluster in clustered_subgraph_split:
                for node in cluster:
                    labels[int(node)] += label
                label += 1
        return labels
    else:
        labels = np.zeros(len(feature_matrix))
        label = 1
        for clusters in clustered_subgraphs:
            for cluster in clusters:
                for node in cluster:
                    labels[int(node)] += label
            label += 1
        return labels