"""
Binary task with an explicit output activation, as requested at the review.

Three conditions, so the two changes can be separated:
    relu_softmax     ReLU hidden, 2 logits, CrossEntropyLoss   (repeats Exp 2)
    relu_sigmoid     ReLU hidden, 1 logit, sigmoid + BCELoss
    leaky_sigmoid    LeakyReLU hidden, 1 logit, sigmoid + BCELoss

An explicit softmax cannot be added before nn.CrossEntropyLoss, which applies
log-softmax internally. The binary task avoids this because sigmoid with BCELoss
is a different objective, not a duplicated one.

Usage: python train_binary_activation.py <config> <seed>
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import math, random, sys, time

CONFIG = sys.argv[1] if len(sys.argv) > 1 else 'leaky_sigmoid'
SEED   = int(sys.argv[2]) if len(sys.argv) > 2 else 42
assert CONFIG in ('relu_softmax', 'relu_sigmoid', 'leaky_sigmoid'), CONFIG
HIDDEN_ACT = 'leakyrelu' if CONFIG.startswith('leaky') else 'relu'
OUTPUT     = 'sigmoid' if CONFIG.endswith('sigmoid') else 'softmax'

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'

X_tr = np.load(f'{D}/X_train_smote_binary.npy')
y_tr = np.load(f'{D}/y_train_smote_binary.npy')
X_te = np.load(f'{D}/X_test_scaled.npy')
y_te = pd.read_csv(f'{D}/y_test_binary.csv').squeeze().values.astype(int)

print(f"Config     : {CONFIG}")
print(f"Hidden act : {HIDDEN_ACT}")
print(f"Output     : {OUTPUT}")
print(f"Device     : {device}   Seed: {SEED}")
print(f"Train {X_tr.shape}   Test {X_te.shape}")
print(f"Attack rate: train {y_tr.mean()*100:.2f}%   test {y_te.mean()*100:.2f}%")
print()

class FlowDataset(Dataset):
    def __init__(self, X, y, float_labels):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if float_labels else torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

FLOAT_LABELS = (OUTPUT == 'sigmoid')
train_loader = DataLoader(FlowDataset(X_tr, y_tr, FLOAT_LABELS), batch_size=512,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
test_loader  = DataLoader(FlowDataset(X_te, y_te, FLOAT_LABELS),
                          batch_size=512, shuffle=False)

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
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        w = F.softmax(sc, dim=-1)
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.W_o(o)

class Layer(nn.Module):
    def __init__(self, d, h, ff=512, dr=0.3, act='relu'):
        super().__init__()
        self.attention = IAAAttention(d, h)
        self.feed_forward = nn.Sequential(
            nn.Linear(d, ff), activation(act), nn.Dropout(dr), nn.Linear(ff, d))
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dr)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class BinaryTransformer(nn.Module):
    def __init__(self, d_model, nhead, layers, dropout=0.3, act='relu', output='softmax'):
        super().__init__()
        self.output = output
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([Layer(d_model, nhead, 512, dropout, act)
                                     for _ in range(layers)])
        n_out = 1 if output == 'sigmoid' else 2
        self.fc = nn.Linear(d_model, n_out)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_projection(x.unsqueeze(2))
        for l in self.layers:
            x = l(x)
        x = self.dropout(x.mean(dim=1))
        z = self.fc(x)
        if self.output == 'sigmoid':
            return torch.sigmoid(z).squeeze(1)
        return z

torch.manual_seed(SEED)
model = BinaryTransformer(128, 4, 3, act=HIDDEN_ACT, output=OUTPUT).to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"Output dim : {1 if OUTPUT == 'sigmoid' else 2}")

criterion = nn.BCELoss() if OUTPUT == 'sigmoid' else nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

EPS = 1e-7
t0 = time.time()
for epoch in range(20):
    model.train()
    ls = 0.0; cor = tot = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        if OUTPUT == 'sigmoid':
            out = out.clamp(EPS, 1 - EPS)
            loss = criterion(out, by)
            cor += ((out > 0.5).long() == by.long()).sum().item()
        else:
            loss = criterion(out, by)
            cor += (out.argmax(1) == by).sum().item()
        loss.backward(); optimizer.step()
        ls += loss.item(); tot += by.size(0)
    print(f"Epoch {epoch+1}/20 - Loss: {ls/len(train_loader):.4f}, "
          f"Accuracy: {100*cor/tot:.2f}%")
train_time = time.time() - t0
print(f"\nTraining time {train_time:.1f}s")

torch.save(model.state_dict(),
           f'/scratch/kdhungel/iads-project/models/binact_{CONFIG}_seed{SEED}.pth')

model.eval()
preds, labels = [], []
t1 = time.time()
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(device))
        if OUTPUT == 'sigmoid':
            preds.extend((out > 0.5).long().cpu().numpy())
            labels.extend(by.long().numpy())
        else:
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(by.numpy())
infer_time = time.time() - t1

NAMES = ['BENIGN', 'ATTACK']
rep = classification_report(labels, preds, target_names=NAMES, digits=4, zero_division=0)
cm  = confusion_matrix(labels, preds)
macro = f1_score(labels, preds, average='macro', zero_division=0)

print(f"\nInference {infer_time:.1f}s for {len(labels):,} samples")
print(f"macro F1  {macro:.4f}")
print("Experiment 2 (relu_softmax, seed 42) gave macro F1 0.9328")
print(rep)
print("Confusion matrix, rows true, cols predicted, labels BENIGN ATTACK")
print(cm)

with open(f'{R}/binact_{CONFIG}_seed{SEED}_report.txt', 'w') as f:
    f.write(f"Binary task, config={CONFIG}, seed={SEED}\n")
    f.write(f"hidden_activation={HIDDEN_ACT}  output_activation={OUTPUT}\n")
    f.write(f"loss={'BCELoss' if OUTPUT == 'sigmoid' else 'CrossEntropyLoss'}\n")
    f.write(f"output_units={1 if OUTPUT == 'sigmoid' else 2}\n")
    f.write(f"train_time_s={train_time:.1f} inference_time_s={infer_time:.1f}\n")
    f.write(f"params={sum(p.numel() for p in model.parameters())}\n")
    f.write(f"macro_f1={macro:.4f}\n\n")
    f.write(rep)
    f.write("\n\nConfusion matrix\nlabels: BENIGN, ATTACK\n")
    f.write(np.array2string(cm, max_line_width=200))

print(f"\nSaved {R}/binact_{CONFIG}_seed{SEED}_report.txt")
