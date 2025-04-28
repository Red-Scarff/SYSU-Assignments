import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from PCA import PCA
from LDA import LDA
from KNN import KNN
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def load_data(mat_path, ins_perclass=11, train_split=9):
    data = io.loadmat(mat_path)
    fea = data['fea']  # shape: (ins_perclass*class_number, dim)
    gnd = data['gnd'].flatten()
    input_dim = fea.shape[1]

    # Reshape for splitting per class
    feat = fea.reshape(-1, ins_perclass, input_dim)
    label = gnd.reshape(-1, ins_perclass)

    # Split train/test per class
    train_data = feat[:, :train_split, :].reshape(-1, input_dim)
    test_data = feat[:, train_split:, :].reshape(-1, input_dim)
    train_label = label[:, :train_split].reshape(-1)
    test_label = label[:, train_split:].reshape(-1)

    return train_data, train_label, test_data, test_label


def show_eigenfaces(components, image_shape=(64,64), title_prefix='PC'):
    # Display first 8 eigenfaces in 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        face = components[:, i]
        # Normalize for display
        img = (face - face.min()) / (face.max() - face.min())
        ax.imshow(img.reshape(*image_shape), cmap='gray')
        ax.set_title(f"{title_prefix} {i+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def scatter_plot(X, y, title='2D Scatter', xlabel='Dim 1', ylabel='Dim 2'):
    classes = np.unique(y)
    plt.figure(figsize=(8, 6))
    for c in classes:
        plt.scatter(X[y==c, 0], X[y==c, 1], label=f'Class {c}', s=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()


def evaluate_knn(X_train, y_train, X_test, y_test, k=5):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = np.mean(y_pred == y_test)
    return acc

# 数据标准化与 SVM 训练
def svm_experiment(X_train, X_test, y_train, y_test):
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 网格搜索优化参数
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    svm = GridSearchCV(SVC(), param_grid, cv=3).fit(X_train_scaled, y_train)
    acc = svm.score(X_test_scaled, y_test)
    return acc

if __name__ == '__main__':
    # Load and split data
    train_X, train_y, test_X, test_y = load_data('Yale_64x64.mat')

    # 1) PCA: train and transform
    pca8 = PCA(n_components=8)
    X_pca8_train = pca8.fit_transform(train_X)
    X_pca8_test = pca8.transform(test_X)

    # Display first 8 eigenfaces
    show_eigenfaces(pca8.components_, image_shape=(64,64), title_prefix='PCA')

    # 2D visualization for PCA
    pca2 = PCA(n_components=2)
    X_pca2_train = pca2.fit_transform(train_X)
    X_pca2_test = pca2.transform(test_X)
    scatter_plot(X_pca2_train, train_y, title='PCA 2D (Train)')
    scatter_plot(X_pca2_test, test_y, title='PCA 2D (Test)')

    # 2) LDA: train and transform
    lda8 = LDA(n_components=8)
    X_lda8_train = lda8.fit_transform(train_X, train_y)
    X_lda8_test = lda8.transform(test_X)

    # Display first 8 LDA directions as images
    show_eigenfaces(lda8.scalings_, image_shape=(64,64), title_prefix='LDA')

    # 2D visualization for LDA
    lda2 = LDA(n_components=2)
    X_lda2_train = lda2.fit_transform(train_X, train_y)
    X_lda2_test = lda2.transform(test_X)
    scatter_plot(X_lda2_train, train_y, title='LDA 2D (Train)', xlabel='LD 1', ylabel='LD 2')
    scatter_plot(X_lda2_test, test_y, title='LDA 2D (Test)', xlabel='LD 1', ylabel='LD 2')

    print("PCA 2D训练数据形状:", X_pca2_train.shape)  # 应为 (训练样本数, 2)
    print("LDA 2D训练数据形状:", X_lda2_train.shape)  # 应为 (训练样本数, 2)

    # 3) KNN evaluation
    acc_pca8 = evaluate_knn(X_pca8_train, train_y, X_pca8_test, test_y, k=5)
    acc_lda8 = evaluate_knn(X_lda8_train, train_y, X_lda8_test, test_y, k=5)

    print(f"KNN accuracy on PCA-reduced (8D) data: {acc_pca8*100:.2f}%")
    print(f"KNN accuracy on LDA-reduced (8D) data: {acc_lda8*100:.2f}%")

    # 4) SVM experiment
    acc_pca_svm = svm_experiment(X_pca8_train, X_pca8_test, train_y, test_y)
    acc_lda_svm = svm_experiment(X_lda8_train, X_lda8_test, train_y, test_y)
    print(f"SVM-PCA Accuracy: {acc_pca_svm*100:.2f}%")
    print(f"SVM-LDA Accuracy: {acc_lda_svm*100:.2f}%")
