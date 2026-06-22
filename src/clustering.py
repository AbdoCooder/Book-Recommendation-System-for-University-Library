import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_users(user_book_matrix: pd.DataFrame, n_clusters: int = 4, n_components: int = 20, random_state: int = 42):
    filled = user_book_matrix.fillna(0)
    n_components = min(n_components, max(2, min(filled.shape) - 1))
    if min(filled.shape) <= 2:
        features = filled.values
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        features = svd.fit_transform(filled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    score = None
    if len(set(labels)) > 1 and len(labels) > n_clusters:
        score = silhouette_score(features, labels)

    clusters = pd.DataFrame({"User-ID": filled.index, "Cluster": labels})
    return clusters, score