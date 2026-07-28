import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import math
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/X_train_clean.csv')
X_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/X_val_clean.csv')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

X_full = pd.concat([X_train, X_val], ignore_index=True)
y_full = pd.concat([y_train, y_val], ignore_index=True)

print(f"X_full shape: {X_full.shape}")

le = LabelEncoder()
y_full_encoded = le.fit_transform(y_full)

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_full_np = X_full.values
y_full_np = y_full_encoded

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_np, y_full_np)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")

    X_tr, X_val_fold = X_full_np[train_idx], X_full_np[val_idx]
    y_tr, y_val_fold = y_full_np[train_idx], y_full_np[val_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val_fold)

    sampling_strategy = {i: 10000 for i in range(15)
                        if np.sum(y_tr == i) < 10000}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr_scaled, y_tr)
    print(f"Training samples after SMOTE: {len(X_tr_resampled)}")

    train_loader = DataLoader(
        NetworkFlowDataset(X_tr_resampled, y_tr_resampled),
        batch_size=512, shuffle=True)
    val_loader = DataLoader(
        NetworkFlowDataset(X_val_scaled, y_val_fold),
        batch_size=512, shuffle=False)

    model = TransformerClassifier(
        input_size=71, d_model=128, nhead=4,
        num_layers=3, num_classes=15).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(10):
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
        print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")

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

    with open(f'/scratch/kdhungel/iads-project/results/transformer_kfold_fold{fold+1}_report.txt', 'w') as f:
        f.write(classification_report(all_labels, all_preds, digits=4))

print("\nTransformer K-Fold Complete!")
