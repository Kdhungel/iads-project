"""
Multi-pass inference, Section 6.4 of the original project proposal.

    pass 1:  logits_1 = f(x);  p = softmax(logits_1)
    pass 2:  bias = alpha * p * log(1/f_c);  logits_2 = logits_1 + bias

No label is read at any point. Reports both passes separately.
Usage:  python train_multipass.py <seed>
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import math, random, sys, time

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'

X_tr = np.load(f'{D}/X_train_smote.npy')
y_tr = np.load(f'{D}/y_train_smote.npy')
X_te = np.load(f'{D}/X_test_scaled.npy')
y_te_raw = pd.read_csv(f'{D}/y_test.csv').squeeze()

src = pd.read_csv(f'{D}/y_train.csv').squeeze()
le = LabelEncoder(); le.fit(src)
y_te = le.transform(y_te_raw)
NAMES = list(le.classes_)
NUM_CLASSES = len(NAMES)

total = len(src)
freq_bias = torch.zeros(NUM_CLASSES)
for i, lab in enumerate(NAMES):
    freq_bias[i] = math.log(total / (src == lab).sum())
freq_bias = freq_bias.to(device)

print(f"Multi-pass inference   seed {SEED}   device {device}")
print(f"Train {X_tr.shape}   Test {X_te.shape}   Classes {NUM_CLASSES}")
for i, lab in enumerate(NAMES):
    print(f"  {lab:<30}{freq_bias[i]:.4f}")
print()

class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X); self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(FlowDataset(X_tr, y_tr), batch_size=512, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(FlowDataset(X_te, y_te), batch_size=512, shuffle=False)

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
    def __init__(self, d, h, ff=512, dr=0.3):
        super().__init__()
        self.attention = IAAAttention(d, h)
        self.feed_forward = nn.Sequential(
            nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dr), nn.Linear(ff, d))
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dr)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class MultiPassTransformer(nn.Module):
    def __init__(self, d_model, nhead, layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([Layer(d_model, nhead, 512, dropout) for _ in range(layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def backbone(self, x):
        x = self.input_projection(x.unsqueeze(2))
        for l in self.layers:
            x = l(x)
        x = self.dropout(x.mean(dim=1))
        return self.fc(x)
    def forward(self, x, freq_bias_all, return_both=False):
        logits1 = self.backbone(x)
        p = F.softmax(logits1, dim=1)
        bias = torch.abs(self.alpha) * p * freq_bias_all
        logits2 = logits1 + bias
        if return_both:
            return logits1, logits2
        return logits2

torch.manual_seed(SEED)
model = MultiPassTransformer(128, 4, 3, NUM_CLASSES).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

t0 = time.time()
for epoch in range(20):
    model.train()
    tot = cor = 0; ls = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx, freq_bias)
        loss = criterion(out, by)
        loss.backward(); optimizer.step()
        ls += loss.item()
        cor += (out.argmax(1) == by).sum().item(); tot += by.size(0)
    print(f"Epoch {epoch+1}/20 - Loss: {ls/len(train_loader):.4f}, "
          f"Accuracy: {100*cor/tot:.2f}%, alpha: {model.alpha.item():.4f}")
train_time = time.time() - t0

print(f"\nTraining time {train_time:.1f}s")
print(f"Learned alpha {model.alpha.item():.4f}")
torch.save(model.state_dict(), f'/scratch/kdhungel/iads-project/models/multipass_seed{SEED}.pth')

model.eval()
p1_all, p2_all, lab_all = [], [], []
t1 = time.time()
with torch.no_grad():
    for bx, by in test_loader:
        l1, l2 = model(bx.to(device), freq_bias, return_both=True)
        p1_all.extend(l1.argmax(1).cpu().numpy())
        p2_all.extend(l2.argmax(1).cpu().numpy())
        lab_all.extend(by.numpy())
infer_time = time.time() - t1

f1_pass1 = f1_score(lab_all, p1_all, average='macro', zero_division=0)
f1_pass2 = f1_score(lab_all, p2_all, average='macro', zero_division=0)
changed = int(np.sum(np.array(p1_all) != np.array(p2_all)))

print(f"\nInference {infer_time:.1f}s for {len(lab_all):,} samples")
print(f"Predictions changed by pass 2: {changed:,} ({100*changed/len(lab_all):.3f}%)")
print(f"Macro F1 pass 1: {f1_pass1:.4f}")
print(f"Macro F1 pass 2: {f1_pass2:.4f}")
print(f"Difference:      {f1_pass2-f1_pass1:+.4f}")

names = le.inverse_transform
rep2 = classification_report(names(lab_all), names(p2_all), digits=4, zero_division=0)
rep1 = classification_report(names(lab_all), names(p1_all), digits=4, zero_division=0)
cm2 = confusion_matrix(lab_all, p2_all)
print("\nPass 2 classification report")
print(rep2)

with open(f'{R}/multipass_seed{SEED}_report.txt', 'w') as f:
    f.write(f"Multi-pass inference, seed={SEED}\n")
    f.write(f"train_time_s={train_time:.1f} inference_time_s={infer_time:.1f}\n")
    f.write(f"learned_alpha={model.alpha.item():.6f}\n")
    f.write(f"predictions_changed_by_pass2={changed} ({100*changed/len(lab_all):.3f}%)\n")
    f.write(f"macro_f1_pass1={f1_pass1:.4f}\n")
    f.write(f"macro_f1_pass2={f1_pass2:.4f}\n\n")
    f.write("PASS 2 (bias applied)\n")
    f.write(rep2)
    f.write("\n\nPASS 1 (backbone only, no bias)\n")
    f.write(rep1)
    f.write("\n\nConfusion matrix, pass 2\nlabels: " + ", ".join(NAMES) + "\n")
    f.write(np.array2string(cm2, max_line_width=250))

print(f"\nSaved {R}/multipass_seed{SEED}_report.txt")
