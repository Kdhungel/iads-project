import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
print(f"Device: {device} — Standard Transformer, Seed: {SEED}")

# Load SMOTE resampled data — same as IAA experiments
X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

print(f"X_train shape: {X_train.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    NetworkFlowDataset(X_train, y_train_encoded),
    batch_size=512, shuffle=True,
    generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(
    NetworkFlowDataset(X_val, y_val_encoded),
    batch_size=512, shuffle=False)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

torch.manual_seed(SEED)
model = TransformerClassifier(
    input_size=71, d_model=128, nhead=4,
    num_layers=3, num_classes=15).to(device)
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
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), f'/scratch/kdhungel/iads-project/models/transformer_seed{SEED}.pth')
print(f"Model saved.")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = le.inverse_transform(all_preds)
all_labels = le.inverse_transform(all_labels)
print(classification_report(all_labels, all_preds, digits=4))

with open(f'/scratch/kdhungel/iads-project/results/transformer_seed{SEED}_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
print("Report saved.")
