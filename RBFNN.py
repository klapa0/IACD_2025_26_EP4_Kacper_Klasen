import numpy as np

from k_means import kmeans
from Adaline import Adaline

class RBFNN:
    def __init__(self, num_hidden_units, eta=0.01, epochs=500):
        self.k = num_hidden_units
        self.eta = eta
        self.epochs = epochs
        self.centroids = None
        self.sigmas = None
        self.adaline = Adaline(eta=self.eta, epochs=self.epochs)
        
    def _gaussian_rbf(self, X):
        Z = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            dist_sq = np.sum((X - self.centroids[i])**2, axis=1)
            Z[:, i] = np.exp(-dist_sq / (2 * self.sigmas[i]**2 + 1e-8))
        return Z
        
    def fit(self, X, y):
        self.centroids, labels = kmeans(X, self.k)
        
        self.sigmas = np.zeros(self.k)
        for i in range(self.k):
            points = X[labels == i]
            if len(points) > 1:
                self.sigmas[i] = np.mean(np.linalg.norm(points - self.centroids[i], axis=1))
            else:
                self.sigmas[i] = 1.0
                
        Z = self._gaussian_rbf(X)
        
        self.adaline.fit(Z, y)
        
    def predict(self, X):
        Z = self._gaussian_rbf(X)
        return self.adaline.predict(Z)
