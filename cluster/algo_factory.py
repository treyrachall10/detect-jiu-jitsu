import yaml

from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, AgglomerativeClustering

def get_cluster_algo(algo):

    with open('config/clustering_yamls/dbscan.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

        if algo == 'DBSCAN':
            params = cfg['clustering']['DBSCAN']
            return DBSCAN(**params)
        elif algo == 'AffinityPropagation':
            params = cfg['clustering']['AffinityPropagation']
            return AffinityPropagation(**params)
        elif algo == 'MeanShift':
            params = cfg['clustering']['MeanShift']
            return MeanShift(**params)
        elif algo == 'Agglomerative':
            params = cfg['clustering']['Agglomerative']
            return AgglomerativeClustering(**params)