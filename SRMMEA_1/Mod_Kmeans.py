from sklearn.cluster import KMeans
import numpy as np


def Kmeans_clu(K, data,C):
    """
    :param K: Number of cluster
    :param data:
    :return:
    """


    kmeans = KMeans(n_clusters=K, init=C, max_iter=1, n_init=1).fit(data)  ##Apply k-means clustering
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    z = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}  #

    return clu_centres, labels