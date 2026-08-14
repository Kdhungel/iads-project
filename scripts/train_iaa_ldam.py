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
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} — LDAM-DRW loss, alpha=0, Seed: {SEED}")

X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val   = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val   = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)
NUM_CLASSES = len(le.classes_)

# ---------------------------------------------------------------------------
# Class counts come from the ORIGINAL distribution, not post-SMOTE.
# LDAM's margin depends on true class rarity; SMOTE counts would erase that.
# ---------------------------------------------------------------------------
y_train_original = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
y_train_original = y_train_original.str.replace('Web Attack \ufffd', 'Web Attack -', regex=False)
cls_num_list = [int((y_train_original == lab).sum()) for lab in le.classes_]
print("Original class counts:")
for lab, n in zip(le.classes_, cls_num_list):
    print(f"  {lab}: {n}")

freq_bias = torch.zeros(NUM_CLASSES)
total_orig = sum(cls_num_list)
for i, n in enumerate(cls_num_list):
    freq_bias[i] = math.log(total_orig / n)
freq_bias = freq_bias.to(device)


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin loss (Cao et al., NeurIPS 2019).

    Core idea: subtract a class-dependent margin from the logit of the TRUE class
    before computing cross entropy. That forces the model to score the true class
    higher than it otherwise would need to, i.e. it demands extra room.

    Margin for class j:  m_j proportional to 1 / n_j^(1/4)
    Fewer samples -> larger n_j^(-1/4) -> larger margin demanded.
    max_m rescales the largest margin to a chosen value.
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_num_list, dtype=np.float64)))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s               # logit scale; LDAM needs scaled logits to work
        self.weight = weight     # optional per-class weights (used by DRW)

    def to(self, device):
        self.m_list = self.m_list.to(device)
        if self.weight is not None:
            self.weight = self.weight.to(device)
        return self

    def forward(self, x, target):
        # one-hot mask marking the true class of each sample
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)

        index_float = index.float()
        # per-sample margin = margin of that sample's true class
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        # subtract margin ONLY from the true-class logit
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NetworkFlowDataset(X_train, y_train_encoded), batch_size=512,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader   = DataLoader(NetworkFlowDataset(X_val, y_val_encoded),
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
model = IAATransformer(71, 128, 4, 3, NUM_CLASSES).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
EPOCHS = 20
DRW_START = 15   # deferred re-weighting kicks in here

for epoch in range(EPOCHS):
    # -----------------------------------------------------------------------
    # DRW: train with plain LDAM first, then switch on class weights near the
    # end. Weighting from epoch 0 destabilises training (we saw this in v6/v7).
    # -----------------------------------------------------------------------
    if epoch < DRW_START:
        weight = None
    else:
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_w = (1.0 - beta) / np.array(effective_num)
        per_cls_w = per_cls_w / np.sum(per_cls_w) * NUM_CLASSES
        weight = torch.FloatTensor(per_cls_w).to(device)
        if epoch == DRW_START:
            print(f"--- DRW active from epoch {epoch+1} ---")

    criterion = LDAMLoss(cls_num_list, max_m=0.3, weight=weight, s=1).to(device)

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
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total_n:.2f}%")

torch.save(model.state_dict(),
           f'/scratch/kdhungel/iads-project/models/iaa_ldam_seed{SEED}.pth')
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

with open(f'/scratch/kdhungel/iads-project/results/iaa_ldam_seed{SEED}_report.txt', 'w') as f:
    f.write(classification_report(labels, preds, digits=4))
print("Report saved.")
