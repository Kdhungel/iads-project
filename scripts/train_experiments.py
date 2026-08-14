"""
Experiments 1-3 from the 11 Aug meeting.

Usage:
    python train_experiments.py exp1     # Leaky ReLU, multiclass
    python train_experiments.py exp2     # binary attack vs benign
    python train_experiments.py exp3     # multiclass, SQL Injection removed

Architecture is the custom IAA-Transformer with alpha fixed at zero, matching
what was reported previously. No attention dropout and default nn.Linear
initialisation are retained deliberately so results stay comparable to the
earlier runs; both are characterised in the mechanism analysis.

Note on the output activation: nn.CrossEntropyLoss applies log-softmax
internally. Adding an explicit softmax before it would apply it twice and
degrade training, so the output layer returns raw logits. Softmax is present,
just inside the loss function.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import math, random, sys, time

EXP = sys.argv[1] if len(sys.argv) > 1 else 'exp1'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'

if EXP == 'exp1':
    X_tr = np.load(f'{D}/X_train_smote.npy')
    y_tr = np.load(f'{D}/y_train_smote.npy')
    X_te = np.load(f'{D}/X_test_scaled.npy')
    y_te_raw = pd.read_csv(f'{D}/y_test.csv').squeeze()
    ACT = 'leakyrelu'
    TAG = 'exp1_leakyrelu'
elif EXP == 'exp2':
    X_tr = np.load(f'{D}/X_train_smote_binary.npy')
    y_tr = np.load(f'{D}/y_train_smote_binary.npy')
    X_te = np.load(f'{D}/X_test_scaled.npy')
    y_te_raw = pd.read_csv(f'{D}/y_test_binary.csv').squeeze()
    ACT = 'relu'
    TAG = 'exp2_binary'
elif EXP == 'exp3':
    X_tr = np.load(f'{D}/X_train_smote_nosql.npy')
    y_tr = np.load(f'{D}/y_train_smote_nosql.npy')
    X_te = np.load(f'{D}/X_test_scaled_nosql.npy')
    y_te_raw = pd.read_csv(f'{D}/y_test_nosql.csv').squeeze()
    ACT = 'relu'
    TAG = 'exp3_nosql'
else:
    raise ValueError(f"unknown experiment: {EXP}")

if EXP == 'exp2':
    y_te = y_te_raw.values.astype(int)
    NAMES = ['BENIGN', 'ATTACK']
else:
    src = pd.read_csv(f'{D}/y_train.csv' if EXP == 'exp1' else f'{D}/y_train_nosql.csv').squeeze()
    le = LabelEncoder(); le.fit(src)
    y_te = le.transform(y_te_raw)
    NAMES = list(le.classes_)

NUM_CLASSES = len(NAMES)
INPUT_SIZE = X_tr.shape[1]

print(f"Experiment : {TAG}")
print(f"Device     : {device}   Seed: {SEED}")
print(f"Activation : {ACT}")
print(f"Train      : {X_tr.shape}   Test: {X_te.shape}")
print(f"Classes    : {NUM_CLASSES}")

class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X); self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(FlowDataset(X_tr, y_tr), batch_size=512, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(FlowDataset(X_te, y_te), batch_size=512, shuffle=False)

def activation(kind):
    return nn.LeakyReLU(0.01) if kind == 'leakyrelu' else nn.ReLU()

class IAAAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
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
        w = F.softmax(scores, dim=-1)
        out = torch.matmul(w, V).transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.W_o(out)

class Layer(nn.Module):
    def __init__(self, d_model, nhead, ff=512, dropout=0.3, act='relu'):
        super().__init__()
        self.attention = IAAAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff), activation(act),
            nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, layers, num_classes, dropout=0.3, act='relu'):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([Layer(d_model, nhead, 512, dropout, act)
                                     for _ in range(layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_projection(x.unsqueeze(2))
        for l in self.layers:
            x = l(x)
        x = self.dropout(x.mean(dim=1))
        return self.fc(x)

torch.manual_seed(SEED)
model = Transformer(128, 4, 3, NUM_CLASSES, act=ACT).to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

t0 = time.time()
for epoch in range(20):
    model.train()
    loss_sum = correct = total = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()
        correct += (out.argmax(1) == by).sum().item()
        total += by.size(0)
    print(f"Epoch {epoch+1}/20 - Loss: {loss_sum/len(train_loader):.4f}, "
          f"Accuracy: {100*correct/total:.2f}%")

train_time = time.time() - t0
print(f"\nTraining time: {train_time:.1f} s ({train_time/60:.1f} min)")

torch.save(model.state_dict(), f'/scratch/kdhungel/iads-project/models/{TAG}.pth')

model.eval()
preds, labels = [], []
t1 = time.time()
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(by.numpy())
infer_time = time.time() - t1

rep = classification_report(labels, preds, target_names=NAMES, digits=4)
cm = confusion_matrix(labels, preds)

print(f"\nInference: {infer_time:.1f} s for {len(labels):,} samples "
      f"({1000*infer_time/len(labels):.4f} ms per sample)")
print("\n" + rep)
print("\nConfusion matrix:")
print(cm)

with open(f'{R}/{TAG}_report.txt', 'w') as f:
    f.write(f"{TAG}  seed={SEED}  activation={ACT}\n")
    f.write(f"train_time_s={train_time:.1f}  inference_time_s={infer_time:.1f}\n")
    f.write(f"params={sum(p.numel() for p in model.parameters())}\n\n")
    f.write(rep)
    f.write("\n\nConfusion matrix\n")
    f.write("labels: " + ", ".join(NAMES) + "\n")
    f.write(np.array2string(cm, max_line_width=200))

np.save(f'{R}/{TAG}_confusion.npy', cm)
print(f"\nSaved: {R}/{TAG}_report.txt")
