import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Dataset
class NetworkFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NetworkFlowDataset(X_train, y_train_encoded), batch_size=512, shuffle=True)
val_loader = DataLoader(NetworkFlowDataset(X_val, y_val_encoded), batch_size=512, shuffle=False)

# Model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = x.unsqueeze(2)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = BiLSTM(input_size=1, hidden_size=128, num_layers=2, num_classes=15).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), '/scratch/kdhungel/iads-project/models/bilstm.pth')
print("Model saved.")

# Evaluate
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

# Save report
with open('/scratch/kdhungel/iads-project/results/bilstm_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
print("Report saved.")
