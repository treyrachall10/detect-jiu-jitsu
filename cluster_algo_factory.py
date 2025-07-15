from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, AgglomerativeClustering

def get_cluster_algo(algo):
    if algo == 'DBSCAN':
        return DBSCAN(eps=.025, min_samples=2)
    elif algo == 'AffinityPropagation':
        return AffinityPropagation(damping=.5)
    elif algo == 'MeanShift':
        return MeanShift(bandwidth=.1)
    elif algo == 'Agglomerative':
        return AgglomerativeClustering(n_clusters=2)