import numpy as np
class KNN:
    """
    k-Nearest Neighbors classifier from scratch.
    """
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            # Compute L2 distances
            dists = np.linalg.norm(self.X_train - x, axis=1)
            # Find k nearest
            idx = np.argsort(dists)[:self.k]
            nearest_labels = self.y_train[idx]
            # Majority vote
            vals, counts = np.unique(nearest_labels, return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        return np.array(y_pred)