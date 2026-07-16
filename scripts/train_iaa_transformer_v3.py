import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

print(f"X_train shape: {X_train.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Load original frequencies before SMOTE
y_train_original = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
total_original = len(y_train_original)
class_freq_original = {}
for label in le.classes_:
    count = (y_train_original == label).sum()
    class_freq_original[label] = count / total_original

freq_bias = torch.zeros(len(le.classes_))
for i, label in enumerate(le.classes_):
    freq_bias[i] = math.log(1.0 / class_freq_original[label])
freq_bias = freq_bias.to(device)

print("Original frequency bias computed.")

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y, freq_bias):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.freq_bias = freq_bias.cpu()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        label = self.y[idx]
        bias = self.freq_bias[label]
        return self.X[idx], label, bias

train_loader = DataLoader(NetworkFlowDataset(X_train, y_train_encoded, freq_bias), batch_size=512, shuffle=True)
val_loader = DataLoader(NetworkFlowDataset(X_val, y_val_encoded, freq_bias), batch_size=512, shuffle=False)

class IAAAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(IAAAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, class_bias):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        bias = torch.abs(self.alpha) * class_bias.view(batch_size, 1, 1, 1)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        return output

class IAATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super(IAATransformerLayer, self).__init__()
        self.attention = IAAAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, class_bias):
        attn_output = self.attention(x, class_bias)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class IAATransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(IAATransformer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            IAATransformerLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, class_bias):
        x = x.unsqueeze(2)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, class_bias)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = IAATransformer(input_size=71, d_model=128, nhead=4, num_layers=3, num_classes=15).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y, batch_bias in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_bias = batch_bias.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, batch_bias)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), '/scratch/kdhungel/iads-project/models/iaa_transformer_v3.pth')
print("Model saved.")

# Print learned alpha values
print("Learned alpha values:")
for name, param in model.named_parameters():
    if 'alpha' in name:
        print(f"  {name}: {param.item():.4f} (abs: {torch.abs(param).item():.4f})")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y, batch_bias in val_loader:
        batch_X = batch_X.to(device)
        batch_bias = batch_bias.to(device)
        outputs = model(batch_X, batch_bias)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = le.inverse_transform(all_preds)
all_labels = le.inverse_transform(all_labels)
print(classification_report(all_labels, all_preds, digits=4))

with open('/scratch/kdhungel/iads-project/results/iaa_transformer_v3_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
print("Report saved.")
EOFcat > scripts/train_iaa_transformer_v3.py << 'EOF'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_train = np.load('/scratch/kdhungel/iads-project/data/processed/X_train_resampled.npy')
y_train = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_resampled.csv').squeeze()
X_val = np.load('/scratch/kdhungel/iads-project/data/processed/X_val_scaled.npy')
y_val = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_val_clean.csv').squeeze()

print(f"X_train shape: {X_train.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Load original frequencies before SMOTE
y_train_original = pd.read_csv('/scratch/kdhungel/iads-project/data/processed/y_train_clean.csv').squeeze()
total_original = len(y_train_original)
class_freq_original = {}
for label in le.classes_:
    count = (y_train_original == label).sum()
    class_freq_original[label] = count / total_original

freq_bias = torch.zeros(len(le.classes_))
for i, label in enumerate(le.classes_):
    freq_bias[i] = math.log(1.0 / class_freq_original[label])
freq_bias = freq_bias.to(device)

print("Original frequency bias computed.")

class NetworkFlowDataset(Dataset):
    def __init__(self, X, y, freq_bias):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.freq_bias = freq_bias.cpu()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        label = self.y[idx]
        bias = self.freq_bias[label]
        return self.X[idx], label, bias

train_loader = DataLoader(NetworkFlowDataset(X_train, y_train_encoded, freq_bias), batch_size=512, shuffle=True)
val_loader = DataLoader(NetworkFlowDataset(X_val, y_val_encoded, freq_bias), batch_size=512, shuffle=False)

class IAAAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(IAAAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, class_bias):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        bias = torch.abs(self.alpha) * class_bias.view(batch_size, 1, 1, 1)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        return output

class IAATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super(IAATransformerLayer, self).__init__()
        self.attention = IAAAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, class_bias):
        attn_output = self.attention(x, class_bias)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class IAATransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(IAATransformer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            IAATransformerLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, class_bias):
        x = x.unsqueeze(2)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, class_bias)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = IAATransformer(input_size=71, d_model=128, nhead=4, num_layers=3, num_classes=15).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y, batch_bias in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_bias = batch_bias.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, batch_bias)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), '/scratch/kdhungel/iads-project/models/iaa_transformer_v3.pth')
print("Model saved.")

# Print learned alpha values
print("Learned alpha values:")
for name, param in model.named_parameters():
    if 'alpha' in name:
        print(f"  {name}: {param.item():.4f} (abs: {torch.abs(param).item():.4f})")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y, batch_bias in val_loader:
        batch_X = batch_X.to(device)
        batch_bias = batch_bias.to(device)
        outputs = model(batch_X, batch_bias)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = le.inverse_transform(all_preds)
all_labels = le.inverse_transform(all_labels)
print(classification_report(all_labels, all_preds, digits=4))

with open('/scratch/kdhungel/iads-project/results/iaa_transformer_v3_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, digits=4))
print("Report saved.")
