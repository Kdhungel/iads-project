import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import math
import random
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} — TCP flags RETAINED, alpha=0, Seed: {SEED}")

# X_train.csv has 78 columns: the 7 sparse TCP flag columns are still present
X_train_raw = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/X_train.csv')
X_val_raw   = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/X_val.csv')
y_train     = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
y_val       = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

y_train = y_train.str.replace('Web Attack \ufffd', 'Web Attack -', regex=False)
y_val   = y_val.str.replace('Web Attack \ufffd', 'Web Attack -', regex=False)

# X_train.csv still contains the 75 negative-duration rows; y_train_clean does not.
# Align by dropping those rows from X so shapes match.
if len(X_train_raw) != len(y_train):
    neg_idx = X_train_raw[X_train_raw['Flow Duration'] < 0].index
    X_train_raw = X_train_raw.drop(index=neg_idx).reset_index(drop=True)
    print(f"Dropped {len(neg_idx)} negative Flow Duration rows to align with labels")

print(f"X_train shape: {X_train_raw.shape}  (flags retained)")
print(f"y_train shape: {y_train.shape}")
assert len(X_train_raw) == len(y_train), "X and y still misaligned"

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)

total = len(y_train)
class_freq = {lab: (y_train == lab).sum() / total for lab in le.classes_}
freq_bias = torch.zeros(len(le.classes_))
for i, lab in enumerate(le.classes_):
    freq_bias[i] = math.log(1.0 / class_freq[lab])
freq_bias = freq_bias.to(device)

sampling_strategy = {i: 10000 for i in range(len(le.classes_))
                     if np.sum(y_train_encoded == i) < 10000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train_encoded)
print(f"Training samples after SMOTE: {len(X_res)}")

INPUT_SIZE = X_train_raw.shape[1]

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NetworkFlowDataset(X_res, y_res), batch_size=512,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader   = DataLoader(NetworkFlowDataset(X_val_scaled, y_val_encoded),
                          batch_size=512, shuffle=False)

class IAAAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x):
        b, s, _ = x.shape
        Q = self.W_q(x).view(b, s, self.nhead, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(b, s, self.nhead, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(b, s, self.nhead, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.W_o(out)

class IAATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.attention = IAAAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class IAATransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            IAATransformerLayer(d_model, nhead, 512, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    def forward(self, x, freq_bias_all):
        x = x.unsqueeze(2)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x + torch.abs(self.alpha) * freq_bias_all

torch.manual_seed(SEED)
model = IAATransformer(INPUT_SIZE, 128, 4, 3, len(le.classes_)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(20):
    model.train()
    running_loss = correct = total_n = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx, freq_bias)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        total_n += by.size(0)
        correct += (pred == by).sum().item()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total_n:.2f}%")

torch.save(model.state_dict(),
           f'/scratch/kdhungel/iads-project/models/iaa_with_flags_seed{SEED}.pth')
print("Model saved.")

model.eval()
preds, labels = [], []
with torch.no_grad():
    for bx, by in val_loader:
        out = model(bx.to(device), freq_bias)
        _, p = torch.max(out, 1)
        preds.extend(p.cpu().numpy())
        labels.extend(by.numpy())

preds  = le.inverse_transform(preds)
labels = le.inverse_transform(labels)
print(classification_report(labels, preds, digits=4))

with open(f'/scratch/kdhungel/iads-project/results/iaa_with_flags_seed{SEED}_report.txt', 'w') as f:
    f.write(classification_report(labels, preds, digits=4))
print("Report saved.")
