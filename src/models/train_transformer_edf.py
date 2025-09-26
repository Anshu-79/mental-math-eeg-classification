import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import mne

sys.path.append(os.path.abspath("."))

from src.models.transformer_model import EEGTransformer
from src.utils.metrics import window_metrics, aggregate_by_file

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_val_subject(train_subjects):
    return train_subjects[-1]  # use last subject as val


def preprocess_raw_edf(edf_path, l_freq=0.5, h_freq=50, resample_sfreq=None):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    # Filter
    data = mne.filter.filter_data(data, sfreq, l_freq=l_freq, h_freq=h_freq)

    # Optional resample
    if resample_sfreq is not None and resample_sfreq != sfreq:
        raw.resample(resample_sfreq)
        data = raw.get_data()
        sfreq = resample_sfreq

    # Normalize per channel
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    return data, sfreq


def create_sequences(data, window_sec, stride_sec, sfreq):
    window_size = int(window_sec * sfreq)
    stride = int(stride_sec * sfreq)
    sequences = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        seq = data[:, start : start + window_size].T  # shape: [time, channels]
        sequences.append(seq)
    return np.stack(sequences)  # shape: [num_seqs, time, channels]


class EEGSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, subjects):
        self.sequences = sequences.astype("float32")
        self.labels = labels.astype("int64")
        self.subjects = np.array(subjects)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        y = torch.tensor(self.labels[idx])
        subj = self.subjects[idx]
        return x, y, subj


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y, subj in loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets, all_subjects = [], [], []
    for x, y, subj in loader:
        x = x.to(device)
        y = y.float().to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.cpu().numpy().tolist())
        all_subjects.extend(subj)
    avg_loss = running_loss / len(loader)
    return avg_loss, all_probs, all_targets, all_subjects


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_dir", type=str, required=True, help="Directory with EDF files")
    parser.add_argument("--subject_csv", type=str, required=True, help="CSV mapping EDF files to subject and condition")
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Sequence length in seconds")
    parser.add_argument("--stride_sec", type=float, default=5.0, help="Stride in seconds")
    parser.add_argument("--resample_sfreq", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints_transformer")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load subject mapping CSV
    subj_df = pd.read_csv(args.subject_csv)  # columns: file_name, subject, condition

    all_sequences, all_labels, all_subjects = [], [], []
    for _, row in subj_df.iterrows():
        file_path = os.path.join(args.edf_dir, row["file_name"])
        data, sfreq = preprocess_raw_edf(file_path, resample_sfreq=args.resample_sfreq)
        sequences = create_sequences(data, args.window_sec, args.stride_sec, sfreq)
        labels = np.full(len(sequences), row["condition"])
        subjects = np.full(len(sequences), row["subject"])
        all_sequences.append(sequences)
        all_labels.append(labels)
        all_subjects.append(subjects)

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_subjects = np.concatenate(all_subjects, axis=0)

    dataset = EEGSequenceDataset(all_sequences, all_labels, all_subjects)
    unique_subjects = sorted(np.unique(all_subjects))
    logging.info(f"Total sequences: {len(dataset)}, subjects: {len(unique_subjects)}")

    all_fold_rows = []

    for fold, test_subject in enumerate(unique_subjects, start=1):
        train_mask = all_subjects != test_subject
        test_mask = all_subjects == test_subject

        train_ds = EEGSequenceDataset(all_sequences[train_mask], all_labels[train_mask], all_subjects[train_mask])
        test_ds = EEGSequenceDataset(all_sequences[test_mask], all_labels[test_mask], all_subjects[test_mask])

        # Validation split from training subjects
        train_subjects = sorted(np.unique(train_ds.subjects))
        val_subject = pick_val_subject(train_subjects)
        val_mask = train_ds.subjects == val_subject
        train_mask_eff = train_ds.subjects != val_subject

        val_ds = EEGSequenceDataset(all_sequences[train_mask][val_mask], all_labels[train_mask][val_mask], all_subjects[train_mask][val_mask])
        train_ds_eff = EEGSequenceDataset(all_sequences[train_mask][train_mask_eff], all_labels[train_mask][train_mask_eff], all_subjects[train_mask][train_mask_eff])

        train_loader = DataLoader(train_ds_eff, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        device = args.device
        input_dim = all_sequences.shape[2]
        model = EEGTransformer(input_dim=input_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        logging.info(f"\n=== Fold {fold} | Test subject: {test_subject} ===")
        logging.info(f"Train: {len(train_ds_eff)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        logging.info(f"Using device: {device}")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(ckpt_dir, f"fold{fold}_best.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, _, _, _ = eval_epoch(model, val_loader, criterion, device)
            logging.info(f"[Fold {fold}] Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({"state_dict": model.state_dict()}, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logging.info(f"[Fold {fold}] Early stopping at epoch {epoch}.")
                    break

        # Load best model and evaluate test
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_probs, test_targs, test_subjs = eval_epoch(model, test_loader, criterion, device)

        wm = window_metrics(np.array(test_targs), np.array(test_probs))

        # Save per-fold results
        for subj, p, t in zip(test_subjs, test_probs, test_targs):
            all_fold_rows.append({
                "fold": fold,
                "test_subject": test_subject,
                "subject": subj,
                "prob": float(p),
                "target": int(t),
            })

        logging.info(f"[Fold {fold}] WINDOW metrics: acc={wm['acc']:.3f} f1={wm['f1']:.3f} auc={wm['roc_auc']:.3f} mcc={wm['mcc']:.3f}")

    # Save summary
    out_csv = os.path.join(args.out_dir, "transformer_edf_results.csv")
    pd.DataFrame(all_fold_rows).to_csv(out_csv, index=False)
    logging.info(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
