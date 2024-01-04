import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(X, y_kmeans, kmeans):
    X = np.array(X)
    fig = plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'orange', 'magenta']
    for i in range(kmeans.n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=150, marker='*', c=colors[i], label='Cluster {}'.format(i+1))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, c='black', marker="o", label='Centroids')
    plt.title('Cluster of Customers', size=20)
    plt.xlabel('Spending Score (1-100)', size=15)
    plt.ylabel('Annual Income (k$)', size=15)
    plt.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
    plt.show()