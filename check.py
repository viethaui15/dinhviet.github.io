from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split

def split_data(X, y_kmeans):
    X_train, X_test, y_train, y_test = train_test_split(X, y_kmeans, test_size=0.3, random_state=125)
    return X_train, X_test, y_train, y_test

def evaluate_model(X_test, y_test, kmeans):
    y_pred = kmeans.predict(X_test)
    ari = adjusted_rand_score(y_pred, y_test)
    ss = silhouette_score(X_test, y_pred)
    return ari, ss