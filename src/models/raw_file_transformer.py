import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

sys.path.append(os.path.abspath("."))

from src.utils.metrics import window_metrics

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ----------------------------
# Dataset
# ----------------------------
class EEGWindowDataset(Dataset):
    def __init__(self, windows, labels, subjects):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # BCE expects float 0/1
        self.subjects = np.array(subjects)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx], self.subjects[idx], idx

# ----------------------------
# EEG Transformer
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class EEGTransformer(nn.Module):
    def __init__(self, n_channels, n_timepoints, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(n_timepoints, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.fc_out(x)
        return logits.squeeze(-1)

# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_val_subject(train_subjects):
    return train_subjects[-1]

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        y = (y - 1).float().to(device)  # ensure 0/1
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets, all_meta = [], [], []
    for x, y, subj, win_idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(-1)
        y = (y - 1).float().to(device)
        loss = criterion(logits, y)
        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.cpu().numpy().tolist())
        all_meta.extend(list(zip(subj, win_idx)))
    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_probs), np.array(all_targets), all_meta

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------
    # Load windows
    # ----------------------
    windows, labels, subjects = [], [], []
    for fpath in sorted(os.listdir(args.windows_dir)):
        if not fpath.endswith(".npz"):
            continue
        dat = np.load(os.path.join(args.windows_dir, fpath), allow_pickle=True)
        n_windows = dat["windows"].shape[0]

        windows.append(dat["windows"])
        file_name = Path(fpath).stem
        condition = int(file_name.split("_")[1])
        labels.extend([condition] * n_windows)
        subjects.extend([file_name.split("_")[0]] * n_windows)

    windows = np.concatenate(windows, axis=0)
    labels = np.array(labels, dtype=np.int64)
    subjects = np.array(subjects)
    logging.info(f"Loaded {len(windows)} windows from {len(np.unique(subjects))} subjects.")

    dataset = EEGWindowDataset(windows, labels, subjects)

    # ----------------------
    # LOSO folds
    # ----------------------
    unique_subjects = sorted(np.unique(subjects))
    all_fold_rows = []
    fold_metrics = []

    for fold, test_subject in enumerate(unique_subjects, start=1):
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        train_dataset = torch.utils.data.Subset(dataset, np.where(train_mask)[0])
        test_dataset = torch.utils.data.Subset(dataset, np.where(test_mask)[0])

        train_subjects = np.unique(subjects[train_mask])
        val_subject = pick_val_subject(train_subjects)

        val_mask = np.array([s == val_subject for s in subjects[train_mask]])
        train_mask_eff = np.array([s != val_subject for s in subjects[train_mask]])

        val_dataset = torch.utils.data.Subset(train_dataset, np.where(val_mask)[0])
        train_dataset = torch.utils.data.Subset(train_dataset, np.where(train_mask_eff)[0])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # ----------------------
        # Model
        # ----------------------
        n_channels, n_timepoints = windows.shape[1], windows.shape[2]
        model = EEGTransformer(n_channels=n_channels, n_timepoints=n_timepoints).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        logging.info(f"\n=== Fold {fold} | Test subject: {test_subject} ===")
        logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(args.out_dir, f"fold{fold}_best.pt")

        # Training loop
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
            val_loss, _, _, _ = eval_epoch(model, val_loader, criterion, args.device)
            logging.info(f"[Fold {fold}] Epoch {epoch} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({"state_dict": model.state_dict()}, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logging.info(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

        # Evaluate test set
        ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_probs, test_targs, test_metas = eval_epoch(model, test_loader, criterion, args.device)

        # Compute metrics
        y_true = test_targs.astype(int)
        y_pred = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        auc = roc_auc_score(y_true, test_probs) if len(np.unique(y_true)) > 1 else float("nan")

        logging.info(f"[Fold {fold}] Test metrics -> ACC: {acc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, ROC-AUC: {auc:.3f}")

        fold_metrics.append({
            "fold": fold,
            "test_subject": test_subject,
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "roc_auc": auc,
            "test_loss": test_loss
        })

        # Save per-window results
        for (subj, win_idx), p, t in zip(test_metas, test_probs, test_targs):
            all_fold_rows.append({
                "fold": fold,
                "test_subject": test_subject,
                "eval_unit": "window",
                "subject": subj,
                "window_idx": win_idx,
                "prob": float(p),
                "target": int(t)
            })

    # Save per-window CSV
    out_csv = os.path.join(args.out_dir, "transformer_raw_windows_results.csv")
    pd.DataFrame(all_fold_rows).to_csv(out_csv, index=False)
    logging.info(f"Saved per-window results to {out_csv}")

    # Save fold metrics CSV
    fold_metrics_csv = os.path.join(args.out_dir, "transformer_raw_windows_fold_metrics.csv")
    pd.DataFrame(fold_metrics).to_csv(fold_metrics_csv, index=False)
    logging.info(f"Saved per-fold metrics to {fold_metrics_csv}")

    # Summary
    summary = pd.DataFrame([{
        "acc": np.mean([m["acc"] for m in fold_metrics]),
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "mcc": np.mean([m["mcc"] for m in fold_metrics]),
        "roc_auc": np.nanmean([m["roc_auc"] for m in fold_metrics])
    }])
    summary_csv = os.path.join(args.out_dir, "transformer_raw_windows_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved overall summary to {summary_csv}")
    logging.info(f"Overall summary:\n{summary.to_string(index=False)}")

if __name__ == "__main__":
    main()
