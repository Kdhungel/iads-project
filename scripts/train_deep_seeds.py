"""
CNN and BiLSTM at three seeds. Closes Dr. Rahman point 6 for the deep baselines.
Usage:  python train_deep_seeds.py <cnn|bilstm> <seed>
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import random, sys, time

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'cnn'
SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'

X_tr = np.load(f'{D}/X_train_smote.npy'); y_tr = np.load(f'{D}/y_train_smote.npy')
X_te = np.load(f'{D}/X_test_scaled.npy')
y_te_raw = pd.read_csv(f'{D}/y_test.csv').squeeze()
src = pd.read_csv(f'{D}/y_train.csv').squeeze()
le = LabelEncoder(); le.fit(src); y_te = le.transform(y_te_raw)
NAMES = list(le.classes_); NC = len(NAMES); NFEAT = X_tr.shape[1]
print(f"{MODEL} seed {SEED}   device {device}   train {X_tr.shape}   test {X_te.shape}")

class FD(Dataset):
    def __init__(s, X, y): s.X = torch.FloatTensor(X); s.y = torch.LongTensor(y)
    def __len__(s): return len(s.X)
    def __getitem__(s, i): return s.X[i], s.y[i]

tl = DataLoader(FD(X_tr, y_tr), batch_size=512, shuffle=True,
                generator=torch.Generator().manual_seed(SEED))
vl = DataLoader(FD(X_te, y_te), batch_size=512, shuffle=False)

class CNN1D(nn.Module):
    def __init__(s, n_feat, n_cls, dr=0.3):
        super().__init__()
        s.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        s.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        s.pool = nn.MaxPool1d(2); s.relu = nn.ReLU(); s.drop = nn.Dropout(dr)
        out = (n_feat // 2) // 2
        s.fc1 = nn.Linear(128 * out, 256); s.fc2 = nn.Linear(256, n_cls)
    def forward(s, x):
        x = x.unsqueeze(1)
        x = s.pool(s.relu(s.conv1(x)))
        x = s.pool(s.relu(s.conv2(x)))
        x = x.flatten(1)
        return s.fc2(s.drop(s.relu(s.fc1(x))))

class BiLSTM(nn.Module):
    def __init__(s, n_cls, hidden=128, layers=2, dr=0.3):
        super().__init__()
        s.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                         bidirectional=True, dropout=dr)
        s.drop = nn.Dropout(dr); s.fc = nn.Linear(hidden * 2, n_cls)
    def forward(s, x):
        o, _ = s.lstm(x.unsqueeze(2))
        return s.fc(s.drop(o[:, -1, :]))

torch.manual_seed(SEED)
model = (CNN1D(NFEAT, NC) if MODEL == 'cnn' else BiLSTM(NC)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

crit = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=0.001)
t0 = time.time()
for ep in range(10):
    model.train(); tot = cor = 0; ls = 0.0
    for bx, by in tl:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad(); o = model(bx); l = crit(o, by)
        l.backward(); opt.step(); ls += l.item()
        cor += (o.argmax(1) == by).sum().item(); tot += by.size(0)
    print(f"Epoch {ep+1}/10 - Loss: {ls/len(tl):.4f}, Accuracy: {100*cor/tot:.2f}%")
tt = time.time() - t0

model.eval(); P, L = [], []
t1 = time.time()
with torch.no_grad():
    for bx, by in vl:
        P.extend(model(bx.to(device)).argmax(1).cpu().numpy()); L.extend(by.numpy())
it = time.time() - t1

rep = classification_report(le.inverse_transform(L), le.inverse_transform(P),
                            digits=4, zero_division=0)
print(f"\nTrain {tt:.1f}s   Inference {it:.1f}s")
print(f"macro F1 {f1_score(L, P, average='macro', zero_division=0):.4f}")
print(rep)

with open(f'{R}/{MODEL}_seed{SEED}_report.txt', 'w') as f:
    f.write(f"{MODEL} seed={SEED}\n")
    f.write(f"train_time_s={tt:.1f} inference_time_s={it:.1f}\n")
    f.write(f"params={sum(p.numel() for p in model.parameters())}\n")
    f.write(f"macro_f1={f1_score(L, P, average='macro', zero_division=0):.4f}\n\n")
    f.write(rep)
    f.write("\n\nConfusion matrix\nlabels: " + ", ".join(NAMES) + "\n")
    f.write(np.array2string(confusion_matrix(L, P), max_line_width=250))
print(f"Saved {R}/{MODEL}_seed{SEED}_report.txt")
