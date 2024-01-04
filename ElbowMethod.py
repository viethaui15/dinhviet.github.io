import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_elbow(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Elbow Method', size=20)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
