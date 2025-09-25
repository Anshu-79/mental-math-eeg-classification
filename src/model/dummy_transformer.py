import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dummy dataset mimicking EEGFeatureDataset
class DummyEEGDataset(Dataset):
    def __init__(self, n_samples=100, n_features=11, seq_len=1):
        self.X = np.random.randn(n_samples, seq_len, n_features).astype("float32")
        self.y = np.random.randint(0, 2, size=(n_samples,)).astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), f"Subj{idx}", idx

# Minimal transformer from your existing code
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class EEGTransformer(nn.Module):
    def __init__(self, input_dim=11, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        logits = self.classifier(x).squeeze(-1)
        return logits

# Training and eval functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y, subj, win in loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []
    for x, y, subj, win in loader:
        x = x.to(device)
        y = y.float().to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.cpu().numpy().tolist())
    return running_loss / len(loader), all_probs, all_targets

# -----------------------------
# Run test
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = DummyEEGDataset(n_samples=100, n_features=11, seq_len=1)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = EEGTransformer(input_dim=11).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
val_loss, probs, targets = eval_epoch(model, loader, criterion, device)

print("Train loss:", train_loss)
print("Val loss:", val_loss)
print("Probs sample:", probs[:5])
print("Targets sample:", targets[:5])
