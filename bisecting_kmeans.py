
# Bisecting K-Means clustering

# Tímea Halmai & Ágoston Czobor
# s1103349    & s1103351


# Importing libraries
import numpy as np


# Helper functions

def convert_to_array(data):
    """
    Converts a list of lists to a 2-D numpy array, to be used in calculations.
    :param data: list of data vectors
    :return: 2D np.array, where the rows contain one element's data
    """
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data, -1)
    return data


def sum_squared_error(data):
    """
    Calculates the sum of squared errors for the given list of data points.
    :param data: data cluster
    :return: the sum of squared errors in the cluster
    """
    arr = convert_to_array(data)
    centroid = np.mean(arr, 0)
    errors = np.linalg.norm(arr - centroid, ord=2, axis=1)
    return np.sum(errors)


# K-Means clustering
# Just a basic kmeans to be used with bisecting kmeans, bisecting can be used with scikit's kmeans as well

def kmeans(data, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using k-means clustering
    algorithm.
    :param data: list of data vectors
    :param k: number of clusters
    :param epochs: number of epochs
    :param max_iter: maximum number of iterations
    :param verbose: if True, prints the SSE of each iteration
    :return: list of clusters, where each cluster is a list of data vectors
    """
    data = convert_to_array(data)
    assert len(data) >= k, "Number of data points can't be less than k"
    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(data)
        centroids = data[0:k, :]
        last_sse = np.inf
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in data:
                index = np.argmin(np.linalg.norm(centroids - p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack([clusters[index], p])
            # Centroid update
            for i in range(k):
                if clusters[i] is None:
                    clusters[i] = np.expand_dims(centroids[i], 0)
                centroids[i] = np.mean(clusters[i], 0)
            # Calculate SSE
            sse = 0
            for i in range(k):
                sse += sum_squared_error(clusters[i])
            if verbose:
                print("Epoch: {}, Iteration: {}, SSE: {}".format(ep, it, sse))
            if last_sse == sse:
                break
            last_sse = sse
        if sse < best_sse:
            best_sse = sse
            best_clusters = clusters
    return best_clusters


# Bisecting K-Means clustering

def bisecting_kmeans(data, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using bisecting k-means
    clustering algorithm.
    :param data: list of data vectors
    :param k: number of clusters
    :param epochs: number of epochs
    :param max_iter: maximum number of iterations
    :param verbose: if True, prints the SSE of each iteration
    :return: list of clusters, where each cluster is a list of data vectors
    """
    data = convert_to_array(data)
    assert len(data) >= k, "Number of data points can't be less than k"
    clusters = [data]
    while len(clusters) < k:
        # Find the cluster with the largest SSE
        max_sse = -np.inf
        max_index = -1
        for i in range(len(clusters)):
            sse = sum_squared_error(clusters[i])
            if sse > max_sse:
                max_sse = sse
                max_index = i
        # Split the cluster with the largest SSE
        cluster = clusters[max_index]
        del clusters[max_index]
        clusters.extend(kmeans(cluster, 2, epochs, max_iter, verbose))
    return clusters