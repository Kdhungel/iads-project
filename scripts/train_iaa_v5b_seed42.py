# Same as v5b but with explicit seed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import math
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, Seed: {SEED}")

X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

y_train_original = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
total_original = len(y_train_original)
class_freq_original = {}
for label in le.classes_:
    count = (y_train_original == label).sum()
    class_freq_original[label] = count / total_original

freq_bias = torch.zeros(len(le.classes_))
for i, label in enumerate(le.classes_):
    freq_bias[i] = math.log(1.0 / class_freq_original[label])
freq_bias = freq_bias.to(device)

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NetworkFlowDataset(X_train, y_train_encoded), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(NetworkFlowDataset(X_val, y_val_encoded), batch_size=512, shuffle=False)

class IAAAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(IAAAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        return output

class IAATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super(IAATransformerLayer, self).__init__()
        self.attention = IAAAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class IAATransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(IAATransformer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            IAATransformerLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, freq_bias_all):
        x = x.unsqueeze(2)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x + torch.abs(self.alpha) * freq_bias_all
        return x

torch.manual_seed(SEED)
model = IAATransformer(input_size=71, d_model=128, nhead=4, num_layers=3, num_classes=15).to(device)
freq_bias_all = freq_bias.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, freq_bias_all)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), f'/scratch/kdhungel/iads-project/models/iaa_v5b_seed{SEED}.pth')
print(f"Model saved. Alpha: {model.alpha.item():.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X, freq_bias_all)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = le.inverse_transform(all_preds)
all_labels = le.inverse_transform(all_labels)
print(classification_report(all_labels, all_preds, digits=4))

with open(f'/scratch/kdhungel/iads-project/results/iaa_v5b_seed{SEED}_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
print("Report saved.")
