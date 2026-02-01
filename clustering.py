
from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(X, k=2):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels

def run_dbscan(X):
    model = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    labels = model.fit_predict(X)
    return labels
