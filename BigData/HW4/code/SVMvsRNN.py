import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import torch.optim as optim

# ================== 参数优化 ==================
num_words = 20000       # 扩大词汇量
maxlen = 200            
tfidf_features = 10000  # 增加特征维度
embed_dim = 256         # 增大词向量维度
hidden_dim = 128        # 增大隐藏层维度
bidirectional = True    # 启用双向LSTM
num_layers = 2          # 增加LSTM层数
dropout = 0.5           
batch_size = 128        # 增大批大小
epochs = 20             
learning_rate = 1e-3    # 明确学习率

# ================== 数据预处理优化 ==================
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # 去除HTML标签
    text = re.sub(r'[^\w\s]', '', text)    # 去除标点符号
    return text.strip()

def load_imdb(data_path):
    texts = []
    labels = []
    label_map = {'pos': 1, 'neg': 0}
    
    for label_name in ['pos', 'neg']:
        dir_path = os.path.join(data_path, label_name)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as f:
                    texts.append(clean_text(f.read()))  # 增加文本清洗
                    labels.append(label_map[label_name])
    return texts, labels

# ================== 改进的序列处理 ==================
class TextProcessor:
    def __init__(self):
        self.word_counts = {}
        self.vocab = []
        self.word_to_idx = {}
        
    def build_vocab(self, texts, num_words):
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
                
        self.vocab = ['<pad>', '<unk>'] + \
                   sorted(self.word_counts, key=self.word_counts.get, reverse=True)[:num_words-2]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
    
    def text_to_sequence(self, text):
        tokens = text.lower().split()[:maxlen]
        return [self.word_to_idx.get(token, 1) for token in tokens]

# ================== 改进的DataLoader ==================
class TextDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        
        # 预先生成序列和长度
        self.sequences = []
        self.lengths = []
        for text in texts:
            seq = processor.text_to_sequence(text)
            self.sequences.append(torch.LongTensor(seq))
            self.lengths.append(len(seq))
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    # 按长度降序排列
    sorted_indices = np.argsort(lengths)[::-1]
    sequences = [sequences[i] for i in sorted_indices]
    labels = torch.FloatTensor([labels[i] for i in sorted_indices])
    lengths = torch.LongTensor([lengths[i] for i in sorted_indices])
    
    # 动态填充
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, labels, lengths

# ================== 改进的RNN模型 ==================
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        # 嵌入层
        x = self.embedding(x)
        
        # 打包序列
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM层
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # 处理双向LSTM
        if bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
            
        # 全连接层
        out = self.dropout(hidden)
        return self.fc(out).squeeze()

# ================== 主程序流程 ==================
if __name__ == "__main__":
    # 数据加载
    train_texts, train_labels = load_imdb('../data/aclImdb/train')
    test_texts, test_labels = load_imdb('../data/aclImdb/test')

    # TF-IDF + SVM（保持不变）
    vectorizer = TfidfVectorizer(max_features=tfidf_features, 
                               stop_words='english', 
                               lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    svm = LinearSVC()
    svm.fit(X_train_tfidf, train_labels)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(test_labels, svm_pred)

    # RNN处理流程
    processor = TextProcessor()
    processor.build_vocab(train_texts, num_words)
    
    train_dataset = TextDataset(train_texts, train_labels, processor)
    test_dataset = TextDataset(test_texts, test_labels, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            collate_fn=collate_fn)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # 改进的训练函数
    def train(model, loader):
        model.train()
        total_loss = 0
        for inputs, labels, lengths in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # 改进的评估函数
    def evaluate(model, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels, lengths in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
        return correct / len(loader.dataset)

    # 训练循环
    best_acc = 0
    for epoch in range(epochs):
        loss = train(model, train_loader)
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # 最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    rnn_acc = evaluate(model, test_loader)

    # 结果输出
    print(f"\nFinal Results:")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"RNN Accuracy: {rnn_acc:.4f}")
