import warnings
from cacham.ElbowMethod import plot_elbow
from cacham.check import split_data, evaluate_model
from cacham.kmeans import train_kmeans
from cacham.loaddata import load_data
from cacham.plot_kmeans import plot_clusters

warnings.filterwarnings("ignore")
X = load_data()
plot_elbow(X)
kmeans, y_kmeans = train_kmeans(X)
print(y_kmeans)
plot_clusters(X, y_kmeans, kmeans)
X_train, X_test, y_train, y_test = split_data(X, y_kmeans)
kmeans.fit(X_train, y_train)
ari, ss = evaluate_model(X_test, y_test, kmeans)
print('Rand index: ', ari)
print('Silhouette score: ', ss)




