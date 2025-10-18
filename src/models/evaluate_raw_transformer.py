import os
import re
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from tqdm import tqdm

# ----------------------------
# Logging setup (ASCII safe)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_log.txt", mode="w", encoding="utf-8", errors="ignore"),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------
# Dataset
# ----------------------------
class EEGWindowDataset(Dataset):
    def __init__(self, windows, labels, subjects):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.subjects = np.array(subjects)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx], self.subjects[idx], idx


# ----------------------------
# EEG Transformer
# ----------------------------
class PositionalEncoding(torch.nn.Module):
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


class EEGTransformer(torch.nn.Module):
    def __init__(self, n_channels, n_timepoints, d_model=64, n_heads=4, n_layers=2, dropout=0.2, num_classes=1):
        super().__init__()
        self.input_fc = torch.nn.Linear(n_timepoints, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_out = torch.nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.fc_out(x)
        return logits.squeeze(-1)


# ----------------------------
# Infer model config
# ----------------------------
def infer_model_config(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    in_features = state_dict["input_fc.weight"].shape[1]
    d_model = state_dict["input_fc.weight"].shape[0]

    if "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
    elif "fc_out.weight" in state_dict:
        num_classes = state_dict["fc_out.weight"].shape[0]
    else:
        num_classes = 1
    return in_features, d_model, num_classes


# ----------------------------
# Evaluation helper
# ----------------------------
@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for x, y, _, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(y.numpy())

    preds = (np.array(all_probs) > 0.5).astype(int)
    targs = np.array(all_targets).astype(int)

    acc = accuracy_score(targs, preds)
    f1 = f1_score(targs, preds)
    try:
        auc = roc_auc_score(targs, all_probs)
    except ValueError:
        auc = np.nan
    mcc = matthews_corrcoef(targs, preds)
    return acc, auc, f1, mcc


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load windows
    logger.info(f"Loading window data from {args.windows_dir}")
    windows, labels, subjects = [], [], []
    for fpath in sorted(os.listdir(args.windows_dir)):
        if not fpath.endswith(".npz"):
            continue
        dat = np.load(os.path.join(args.windows_dir, fpath), allow_pickle=True)
        n_windows = dat["windows"].shape[0]
        windows.append(dat["windows"])
        name = Path(fpath).stem
        try:
            condition = int(name.split("_")[1])
        except:
            raise ValueError(f"Cannot parse label from {name}")
        labels.extend([condition] * n_windows)
        subjects.extend([name.split("_")[0]] * n_windows)

    windows = np.concatenate(windows)
    labels = np.array(labels)
    subjects = np.array(subjects)
    unique_subjects = sorted(np.unique(subjects))
    logger.info(f"Loaded {len(windows)} windows from {len(unique_subjects)} subjects.")

    # Evaluate all checkpoints
    ckpts = sorted(
        [os.path.join(args.checkpoints_dir, f)
         for f in os.listdir(args.checkpoints_dir)
         if f.startswith("fold") and f.endswith("_best.pt")]
    )
    logger.info(f"Found {len(ckpts)} trained fold models to evaluate.")
    all_results = []

    for i, ckpt_path in enumerate(tqdm(ckpts, desc="Evaluating folds")):
        fold_num = int(re.search(r"fold(\d+)_best", ckpt_path).group(1))
        if fold_num > len(unique_subjects):
            logger.warning(f"Fold {fold_num} exceeds subject count. Skipping.")
            continue

        test_subject = unique_subjects[fold_num - 1]
        test_mask = subjects == test_subject
        if not np.any(test_mask):
            logger.warning(f"Missing data for fold{fold_num}, skipping.")
            continue

        test_dataset = torch.utils.data.Subset(dataset := EEGWindowDataset(windows, labels, subjects),
                                               np.where(test_mask)[0])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Infer model dimensions
        in_features, d_model, num_classes = infer_model_config(ckpt_path)
        n_channels, n_timepoints = windows.shape[1], windows.shape[2]

        # Transpose dynamically if needed
        if in_features == n_channels:
            logger.info(f"[Fold {fold_num}] Using windows as (batch, channels={n_channels}, timepoints={n_timepoints}).")
        elif in_features == n_timepoints:
            logger.info(f"[Fold {fold_num}] Detected transposed training format — swapping axes.")
            windows = np.transpose(windows, (0, 2, 1))
            n_channels, n_timepoints = windows.shape[1], windows.shape[2]
        else:
            logger.warning(f"[Fold {fold_num}] Checkpoint expects {in_features} features. Proceeding with flexible load.")

        model = EEGTransformer(n_channels, n_timepoints, d_model, num_classes=num_classes).to(args.device)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        state_dict = ckpt.get("state_dict", ckpt)

        # Rename layers if needed
        if "fc.weight" in state_dict and "fc_out.weight" not in state_dict:
            state_dict["fc_out.weight"] = state_dict.pop("fc.weight")
            state_dict["fc_out.bias"] = state_dict.pop("fc.bias")

        # Flexible load
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"[Fold {fold_num}] Loaded model. Missing keys: {missing}, Unexpected keys: {unexpected}")

        # Evaluate
        acc, auc, f1, mcc = eval_model(model, test_loader, args.device)
        logger.info(f"[Fold {fold_num}] ACC={acc:.3f}, AUC={auc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}")
        all_results.append({"fold": fold_num, "subject": test_subject, "ACC": acc, "AUC": auc, "F1": f1, "MCC": mcc})

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.checkpoints_dir, "transformer_raw_windows_metrics.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-fold metrics saved to {results_path}")

    if len(results_df) > 0:
        summary = results_df[["ACC", "AUC", "F1", "MCC"]].mean().to_dict()
        summary_path = os.path.join(args.checkpoints_dir, "transformer_raw_windows_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
    else:
        logger.warning("No results available — all folds skipped.")


if __name__ == "__main__":
    main()
