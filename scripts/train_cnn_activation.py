"""
CNN with the activation placement described at the review: an activation applied
at the convolutional stage and an explicit activation at the output layer.

Mirrors train_binary_activation.py so the two model families are directly
comparable. Binary task, same 80/20 split, same SMOTE-resampled training data.

    relu_softmax     ReLU at conv, 2 logits, CrossEntropyLoss   (matches Sec 7.4 CNN)
    relu_sigmoid     ReLU at conv, 1 logit, sigmoid + BCELoss
    leaky_sigmoid    LeakyReLU at conv, 1 logit, sigmoid + BCELoss

Architecture is the 1D-CNN from notebook 05: Conv1d 1 -> 64 -> 128 with kernel 3
and padding 1, MaxPool1d(2) after each block, then FC 256 and the output layer.

Usage:  python train_cnn_activation.py <config> <seed>
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import random, sys, time

CONFIG = sys.argv[1] if len(sys.argv) > 1 else 'leaky_sigmoid'
SEED   = int(sys.argv[2]) if len(sys.argv) > 2 else 42
assert CONFIG in ('relu_softmax', 'relu_sigmoid', 'leaky_sigmoid'), CONFIG
CONV_ACT = 'leakyrelu' if CONFIG.startswith('leaky') else 'relu'
OUTPUT   = 'sigmoid' if CONFIG.endswith('sigmoid') else 'softmax'

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'

X_tr = np.load(f'{D}/X_train_smote_binary.npy')
y_tr = np.load(f'{D}/y_train_smote_binary.npy')
X_te = np.load(f'{D}/X_test_scaled.npy')
y_te = pd.read_csv(f'{D}/y_test_binary.csv').squeeze().values.astype(int)

print(f"Model      : 1D-CNN")
print(f"Config     : {CONFIG}")
print(f"Conv act   : {CONV_ACT}")
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

class CNN1D(nn.Module):
    def __init__(self, n_feat, dropout=0.3, act='relu', output='softmax'):
        super().__init__()
        self.output = output
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.act1  = activation(act)       # activation at the conv stage
        self.act2  = activation(act)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(dropout)
        out_len = (n_feat // 2) // 2
        self.fc1 = nn.Linear(128 * out_len, 256)
        self.act3 = activation(act)
        n_out = 1 if output == 'sigmoid' else 2
        self.fc2 = nn.Linear(256, n_out)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.flatten(1)
        x = self.drop(self.act3(self.fc1(x)))
        z = self.fc2(x)
        if self.output == 'sigmoid':
            return torch.sigmoid(z).squeeze(1)   # explicit output activation
        return z                                  # softmax lives in the loss

torch.manual_seed(SEED)
model = CNN1D(X_tr.shape[1], act=CONV_ACT, output=OUTPUT).to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"Output dim : {1 if OUTPUT == 'sigmoid' else 2}")

criterion = nn.BCELoss() if OUTPUT == 'sigmoid' else nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPS = 1e-7
t0 = time.time()
for epoch in range(10):
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
    print(f"Epoch {epoch+1}/10 - Loss: {ls/len(train_loader):.4f}, "
          f"Accuracy: {100*cor/tot:.2f}%")
train_time = time.time() - t0
print(f"\nTraining time {train_time:.1f}s")

torch.save(model.state_dict(),
           f'/scratch/kdhungel/iads-project/models/cnnact_{CONFIG}_seed{SEED}.pth')

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
print(rep)
print("Confusion matrix, rows true, cols predicted, labels BENIGN ATTACK")
print(cm)
print(f"\nMissed attacks: {cm[1][0]:,} of {cm[1].sum():,}")
print(f"False positives: {cm[0][1]:,} of {cm[0].sum():,}")

with open(f'{R}/cnnact_{CONFIG}_seed{SEED}_report.txt', 'w') as f:
    f.write(f"1D-CNN binary task, config={CONFIG}, seed={SEED}\n")
    f.write(f"conv_activation={CONV_ACT}  output_activation={OUTPUT}\n")
    f.write(f"loss={'BCELoss' if OUTPUT == 'sigmoid' else 'CrossEntropyLoss'}\n")
    f.write(f"output_units={1 if OUTPUT == 'sigmoid' else 2}\n")
    f.write(f"train_time_s={train_time:.1f} inference_time_s={infer_time:.1f}\n")
    f.write(f"params={sum(p.numel() for p in model.parameters())}\n")
    f.write(f"macro_f1={macro:.4f}\n")
    f.write(f"missed_attacks={cm[1][0]} false_positives={cm[0][1]}\n\n")
    f.write(rep)
    f.write("\n\nConfusion matrix\nlabels: BENIGN, ATTACK\n")
    f.write(np.array2string(cm, max_line_width=200))

print(f"\nSaved {R}/cnnact_{CONFIG}_seed{SEED}_report.txt")
