import numpy as np


def k_means(data, k, max_iterations=100, tolerance=1e-4):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    closest_centroids = np.zeros(data.shape[0], dtype=np.int)
    for iteration in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        closest_centroids = np.argmin(distances, axis=0)

        new_centroids = np.array([data[closest_centroids == i].mean(axis=0) for i in range(k)])

        if np.all(np.abs(new_centroids - centroids) <= tolerance):
            break

        centroids = new_centroids

    return centroids, closest_centroids
