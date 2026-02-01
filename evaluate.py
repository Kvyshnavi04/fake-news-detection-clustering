
from sklearn.metrics import adjusted_rand_score, silhouette_score

def evaluate_clustering(true_labels, pred_labels, X):
    ari = adjusted_rand_score(true_labels, pred_labels)
    silhouette = silhouette_score(X, pred_labels)

    print(f"Adjusted Rand Index: {ari}")
    print(f"Silhouette Score: {silhouette}")
