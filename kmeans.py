from sklearn.cluster import KMeans

def train_kmeans(X):
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans, y_kmeans