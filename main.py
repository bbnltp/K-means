
# Data Mining Project - Main file

# Tímea Halmai & Ágoston Czobor
# s1103349    & s1103351

# data from https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering @ 2023.01.02.


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import bisecting_kmeans as bk


def visualize_clusters(clusters, features=None, diminds=None, title=None, save=True):
    """
    Visualizes the 2 dimensions that are specified in diminds.
    :param clusters: list of clusters, where each cluster is a list of data vectors
    :param diminds: list of indices of dimensions to be visualized
    :return: None
    """
    if diminds is None:
        diminds = [0, 1]
    plt.figure()
    for cluster in clusters:
        points = bk.convert_to_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:, diminds[0]], points[:, diminds[1]], 'o')
    if features is not None:
        plt.xlabel(features[diminds[0]])
        plt.ylabel(features[diminds[1]])

    if not save:
        plt.show()
    else:
        if title is None:
            plt.savefig(features[0] + " " + features[1] + ".png")
        else:
            plt.savefig(title + ".png")



if __name__ == '__main__':
    # Importing the dataset
    dataset = pd.read_csv('wine-clustering.csv')
    print(dataset.head())

    # Chose the features to be used
    # features = dataset.columns.values # all features
    features = ['Malic_Acid', 'Hue']

    # Get the data of the specified features
    X = np.array(dataset[features].values.tolist())

    # Use the bisecting k-means algorithm to cluster the data
    # culsters are going to contain "features.shape" dimensional data vectors
    clusters = bk.bisecting_kmeans(X, k=3, epochs=10, max_iter=100, verbose=False)
    #print(clusters)

    # Visualize the clusters on some dimensions
    visualize_clusters(clusters, features=features)
    #visualize_clusters(clusters, diminds=[0, 2])

    #compare kmeans with bisecting kmeans
    # features = dataset.columns.values
    k = 3
    epoch = 1
    max_iter = 20
    start_time = time.time()
    bk.kmeans(X, k=k, epochs=epoch, max_iter=max_iter, verbose=False)
    print("kmeans time: ", time.time() - start_time)
    start_time = time.time()
    bk.bisecting_kmeans(X, k=k, epochs=epoch, max_iter=max_iter, verbose=False)
    print("bisecting kmeans time: ", time.time() - start_time)

    # Outlier test
    # injecting outlier
    X[0][1] *= 10
    clusters = bk.bisecting_kmeans(X, k=3, epochs=10, max_iter=100, verbose=False)
    visualize_clusters(clusters, features=features, title="outlierbk")
    clusters = bk.kmeans(X, k=3, epochs=10, max_iter=100, verbose=False)
    visualize_clusters(clusters, features=features, title="outlierrk")

