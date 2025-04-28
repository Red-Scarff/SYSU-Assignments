import numpy as np
class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.scalings_ = None
        self.means_ = {}
        self.overall_mean_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        n_features = X.shape[1]
        self.overall_mean_ = np.mean(X, axis=0)
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.means_[c] = mean_c
            S_w += (X_c - mean_c).T @ (X_c - mean_c)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - self.overall_mean_).reshape(-1, 1)
            S_b += n_c * (mean_diff @ mean_diff.T)

        # 添加正则项解决奇异性
        S_w += 1e-4 * np.eye(S_w.shape[0])
        # 使用伪逆提高稳定性
        S_w_pinv = np.linalg.pinv(S_w)
        eigvals, eigvecs = np.linalg.eig(S_w_pinv @ S_b)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = np.argsort(eigvals)[::-1]
        self.scalings_ = eigvecs[:, idx[:self.n_components]]
        return self

    def transform(self, X):
        X_centered = X - self.overall_mean_
        return X_centered @ self.scalings_
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)