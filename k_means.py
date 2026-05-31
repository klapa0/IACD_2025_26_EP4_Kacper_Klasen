import numpy as np


def kmeans(X, k, max_iters=100):
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
                
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, labels
