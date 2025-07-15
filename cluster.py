import ast
import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import reduce_feature_list_normalize, print_label_cluster_pairs

def cluster(algo):
    print("Clustering features...")

    df = pd.read_csv(f"csv_output/has_features/features_osnet_x1_0_bjjvideo.csv")

    # Get features and their image they correspond to
    x = reduce_feature_list_normalize(df=df)
    images = df['filename'].to_numpy()

    # Cluster features 
    algo.fit(x)
    labels = algo.labels_

    # Creates new array of same shape as labels, all items are false
    core_samples_mask = np.zeros_like(labels, dtype=bool)

    # Counts number of clusters
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Counts number of unique labels
    unique_labels = set(labels)
    

    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Iterate through unique labels and every color
    for k, col in zip(unique_labels, colors):

        # Assign color black if label is an outlier
        if k == -1:
            
            col = (0.0, 0.0, 0.0, 1.0)

        class_member_mask = (labels == k)
    
        # Plots core samples
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=6)

        # Plots border points
        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=6)
        
        print_label_cluster_pairs(member_mask=class_member_mask, images=images, k=k, color=col)
        
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id:>3} → {count} point(s)")

    print("Finished clustering features...")
    
    plt.title('number of clusters: %d' % n_clusters_)
    plt.show()