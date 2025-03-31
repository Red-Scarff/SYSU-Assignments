import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score
from scipy.stats import mode
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import rcParams
import warnings
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载并预处理数据
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 可视化函数
def plot_clusters(X_reduced, labels, title, noise_mask=None, save_path=None):
    plt.figure(figsize=(10, 8))
    if noise_mask is not None:
        # 绘制噪声点
        plt.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1], 
                    c='gray', s=15, label='噪声点', alpha=0.6)
        # 绘制簇点
        cluster_labels = labels[~noise_mask]
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            plt.scatter(X_reduced[~noise_mask][cluster_mask, 0], 
                        X_reduced[~noise_mask][cluster_mask, 1], 
                        label=f'簇 {cluster}', cmap='viridis', s=50)
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            plt.scatter(X_reduced[label_mask, 0], X_reduced[label_mask, 1], 
                        label=f'簇 {label}', cmap='viridis', s=50)
    
    plt.title(title, fontsize=14)
    plt.xlabel('主成分1', fontsize=12)
    plt.ylabel('主成分2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 忽略特定的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# 优化后的K-means实现
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # 改进的质心初始化：使用更稳定的随机选择
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # 使用cdist加速距离计算
            distances = cdist(X, self.centroids, 'euclidean')
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                      for i in range(self.n_clusters)])
            
            # 处理空簇问题
            empty_clusters = np.where(np.isnan(new_centroids).any(axis=1))
            for i in empty_clusters:
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
            
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

# 优化后的DBSCAN实现
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        n = X.shape[0]
        self.labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        dist_matrix = cdist(X, X, 'euclidean')
        
        for i in range(n):
            if self.labels[i] != -1:
                continue
                
            neighbors = np.where(dist_matrix[i] < self.eps)[0]
            if len(neighbors) < self.min_samples:
                self.labels[i] = -2  # 噪声点
            else:
                self._expand_cluster(i, neighbors, cluster_id, dist_matrix)
                cluster_id += 1

    def _expand_cluster(self, point, neighbors, cluster_id, dist_matrix):
        self.labels[point] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels[neighbor] == -2:
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
                new_neighbors = np.where(dist_matrix[neighbor] < self.eps)[0]
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.unique(np.concatenate([neighbors, new_neighbors]))
            i += 1


def evaluate_clustering(X, true_labels, pred_labels, algorithm_name):
    # 创建有效数据掩码
    valid_mask = pred_labels >= 0
    X = X[valid_mask]
    true_labels = true_labels[valid_mask]
    pred_labels = pred_labels[valid_mask]

    # 标签对齐（匈牙利算法）
    def _align_clusters(y_true, y_pred):
        contingency = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-contingency)
        return col_ind[y_pred]
    
    aligned_labels = _align_clusters(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, aligned_labels)

    # 计算轮廓系数（带有效性检查）
    sc = silhouette_score(X, pred_labels) if len(np.unique(pred_labels))>1 else np.nan

    # 计算CH指数（带保护机制）
    ch = calinski_harabasz_score(X, pred_labels) if len(np.unique(pred_labels))>1 else np.nan

    # 打印结果时增加异常值标记
    print(f"{algorithm_name} 评估结果:")
    print(f"  有效样本数: {len(X)}")
    print(f"  准确率: {accuracy:.4f}{' (对齐后)' if algorithm_name!='True Labels' else ''}")
    print(f"  轮廓系数: {sc:.4f}{'' if not np.isnan(sc) else ' (无效值)'}")
    print(f"  Calinski-Harabasz指数: {ch:.4f}{'' if not np.isnan(ch) else ' (无效值)'}")
    print()
    
    return accuracy, sc, ch


# 主程序
def main():
    os.makedirs('../results', exist_ok=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # print(y)
    # K-means实验
    kmeans_params = [2, 3, 4]
    for k in kmeans_params:
        model = KMeans(n_clusters=k)
        model.fit(X_scaled)
        evaluate_clustering(X_scaled, y, model.labels, f"K-means (k={k})")
        plot_clusters(X_pca, model.labels, f"K-means聚类结果 (k={k})", save_path=f"../results/kmeans_k{k}.png")
    
    # DBSCAN实验
    dbscan_params = [(0.5, 5), (1.0, 10), (1.5, 15)]
    for eps, min_samples in dbscan_params:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X_scaled)
        valid_mask = model.labels != -2
        evaluate_clustering(X_scaled[valid_mask], y[valid_mask], 
                           model.labels[valid_mask], 
                           f"DBSCAN (eps={eps}, min_samples={min_samples})")
        plot_clusters(X_pca, model.labels, 
                     f"DBSCAN聚类结果 (eps={eps}, min_samples={min_samples})",
                     noise_mask=(model.labels == -2), save_path=f"../results/dbscan_eps{eps}_min{min_samples}.png")

if __name__ == "__main__":
    main()