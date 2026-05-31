import numpy as np

class Adaline:
    def __init__(self, eta=0.01, epochs=500):
        self.eta = eta
        self.epochs = epochs
        self.w = None
        self.b = None
        self.mse_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.normal(0, 0.01, n_features)
        self.b = 0.0
        
        for _ in range(self.epochs):
            net_input = np.dot(X, self.w) + self.b
            errors = y - net_input
            
            self.w += self.eta * (2.0 / n_samples) * np.dot(X.T, errors)
            self.b += self.eta * (2.0 / n_samples) * np.sum(errors)
            
            mse = np.mean(errors**2)
            self.mse_history.append(mse)
            
    def predict(self, X):
        return np.dot(X, self.w) + self.b
