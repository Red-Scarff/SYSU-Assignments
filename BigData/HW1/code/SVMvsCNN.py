import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################
# 统一数据预处理（CNN和SVM使用相同的预处理）
#########################################################
shared_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 统一归一化
])

# 加载完整CIFAR-10数据集
full_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=shared_transform)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=shared_transform)

#########################################################
# 创建相同的数据子集（20%训练数据，20%测试数据）
#########################################################
def create_subset(dataset, sample_ratio=0.2):
    num_samples = int(len(dataset) * sample_ratio)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)

# 创建训练子集和测试子集
train_subset = create_subset(full_dataset, sample_ratio=0.2)
test_subset = create_subset(test_dataset, sample_ratio=0.2)

#########################################################
# 特征提取函数（CNN和SVM使用相同的数据）
#########################################################
def extract_features(dataset):
    features = []
    labels = []
    for img, label in dataset:
        features.append(img.numpy().flatten())  # 展平为3072维向量
        labels.append(label)
    return np.array(features), np.array(labels)

print("正在提取特征...")
X_train, y_train = extract_features(train_subset)
X_test, y_test = extract_features(test_subset)

#########################################################
# SVM模型
#########################################################
print("\n===== 训练SVM模型 =====")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', verbose=True)

start_time = time.time()
svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start_time

start_time = time.time()
svm_pred = svm_model.predict(X_test)
svm_test_time = time.time() - start_time

svm_acc = accuracy_score(y_test, svm_pred) * 100

#########################################################
# CNN模型（保持原始结构，使用相同子集）
#########################################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建数据加载器
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print("\n===== 训练CNN模型 =====")
cnn_train_times = []
cnn_test_acc = []

for epoch in range(20):
    # 训练
    start_time = time.time()
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    cnn_train_times.append(time.time() - start_time)

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    cnn_test_acc.append(acc)
    print(f'Epoch [{epoch+1}/20] Test Acc: {acc:.2f}%')

cnn_final_acc = cnn_test_acc[-1]
cnn_total_train_time = sum(cnn_train_times)

#########################################################
# 结果对比
#########################################################
print("\n===== 性能对比 =====")
print(f"数据集大小 | 训练样本: {len(train_subset)} | 测试样本: {len(test_subset)}")
print(f"SVM 测试准确率: {svm_acc:.2f}% | 训练时间: {svm_train_time:.1f}s")
print(f"CNN 测试准确率: {cnn_final_acc:.2f}% | 总训练时间: {cnn_total_train_time:.1f}s")

# 绘制对比曲线
plt.figure(figsize=(10, 5))
plt.plot(cnn_test_acc, label='CNN Test Accuracy')
plt.axhline(y=svm_acc, color='r', linestyle='--', label='SVM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Comparison on Identical Subset (20% Training Data)')
plt.legend()
plt.savefig('../results/SVMvsCNN.png')
plt.show()