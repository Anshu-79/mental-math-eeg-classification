import os
import sys
import json
import math
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut

from src.data.datasets import (
    scan_windows_dir,
    parse_subj_cond,
    compute_channel_zstats,
    EEGWindowsDataset,
)
from src.models.cnn_lstm import CNN_LSTM
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
    # simple heuristic: use the last one as validation subject
    return train_subjects[-1]


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    probs, targets, metas = [], [], []
    for x, y, m in loader:
        # move tensors to device
        x = x.to(device)
        y = y.float().to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.tolist())
        targets.extend(y.cpu().numpy().astype(int).tolist())

        # --- normalize batch metadata `m` into a list of dicts ---
        batch_meta_list = []
        if isinstance(m, dict):
            # m: dict of lists -> convert to list of dicts
            # Get length from any value
            any_val = next(iter(m.values()))
            batch_len = len(any_val)
            keys = list(m.keys())
            for i in range(batch_len):
                entry = {}
                for k in keys:
                    # handle numpy arrays or tensors inside the dict values
                    v = m[k][i]
                    # if v is a 0-d numpy array or tensor, convert to python scalars/strings
                    if hasattr(v, "item"):
                        try:
                            v = v.item()
                        except Exception:
                            pass
                    batch_meta_list.append if False else None  # noop to satisfy linters
                # build entry properly (do in second loop to avoid repeated append bug)
            # The above was just to check; rebuild below more clearly
            batch_meta_list = []
            keys = list(m.keys())
            # handle case where values are tensors/numpy arrays/py lists
            vals_lists = {k: list(m[k]) for k in keys}
            batch_len = len(vals_lists[keys[0]]) if keys else 0
            for i in range(batch_len):
                entry = {k: vals_lists[k][i] for k in keys}
                batch_meta_list.append(entry)

        elif isinstance(m, (list, tuple)):
            # m is already a list/tuple of metadata dicts or scalars
            # if it's list of dicts, keep as-is
            # if it's list of scalars (filepaths), try to wrap them into dicts
            if len(m) == 0:
                batch_meta_list = []
            else:
                first = m[0]
                if isinstance(first, dict):
                    batch_meta_list = list(m)
                else:
                    # assume it's a list of file paths or scalars -> wrap into dict
                    batch_meta_list = [{"file_path": str(mm)} for mm in m]
        else:
            # unexpected type: try to coerce single meta item into list
            try:
                batch_meta_list = [dict(m)]
            except Exception:
                # fallback: represent as string
                batch_meta_list = [{"file_path": str(m)}]

        # finally extend metas
        metas.extend(batch_meta_list)

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else math.nan
    return avg_loss, probs, targets, metas


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows_dir",
        type=str,
        required=True,
        help="Path to data/derivatives/raw_windows/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports",
        help="Where to store csvs and checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Enable train-time augmentations"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints_cnn_lstm")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Scan window files
    items = scan_windows_dir(args.windows_dir)
    df_items = pd.DataFrame(items)
    subjects = sorted(df_items["subject"].unique().tolist())
    logging.info(f"Found {len(items)} files from {len(subjects)} subjects.")

    # LOSO
    logo = LeaveOneGroupOut()
    groups = df_items["subject"].values
    file_paths = df_items["path"].values
    file_subjects = df_items["subject"].values

    all_fold_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(
        logo.split(file_paths, file_subjects, file_subjects), start=1
    ):
        test_subj = file_subjects[test_idx][0]
        train_files = file_paths[train_idx].tolist()
        test_files = file_paths[test_idx].tolist()

        # choose validation subject from train subjects
        train_subjects = sorted(df_items.iloc[train_idx]["subject"].unique().tolist())
        val_subj = pick_val_subject(train_subjects)
        val_mask = [val_subj == parse_subj_cond(Path(fp).stem)[0] for fp in train_files]
        val_files = [fp for fp, m in zip(train_files, val_mask) if m]
        train_files_eff = [fp for fp, m in zip(train_files, val_mask) if not m]

        logging.info(
            f"\n=== Fold {fold_i} | Test subject: {test_subj} | Val subject: {val_subj} ==="
        )
        logging.info(
            f"Train files: {len(train_files_eff)} | Val files: {len(val_files)} | Test files: {len(test_files)}"
        )

        # Fit z-scoring on TRAIN files only
        z_mean, z_std = compute_channel_zstats(train_files_eff)

        # Build datasets/loaders
        train_ds = EEGWindowsDataset(
            train_files_eff, z_mean=z_mean, z_std=z_std, augment=args.augment
        )
        val_ds = EEGWindowsDataset(val_files, z_mean=z_mean, z_std=z_std, augment=False)
        test_ds = EEGWindowsDataset(
            test_files, z_mean=z_mean, z_std=z_std, augment=False
        )

        # infer shapes
        sample_x, _, _ = train_ds[0]
        C, T = sample_x.shape

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        # Model/optim
        device = args.device
        model = CNN_LSTM(
            n_channels=C, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.3
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        logging.info(f"Using device: {device}")

        # Training w/ early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(ckpt_dir, f"fold{fold_i}_best.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            val_loss, _, _, _ = eval_epoch(model, val_loader, device, criterion)
            logging.info(
                f"[Fold {fold_i}] Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}"
            )

            # LR scheduler (reduce on plateau)
            # simple manual: reduce LR by 0.5 if no improve 3 epochs
            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({"state_dict": model.state_dict()}, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve % 3 == 0:
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["lr"] * 0.5
                    logging.info(
                        f"[Fold {fold_i}] Reducing LR to {optimizer.param_groups[0]['lr']:.2e}"
                    )
                if epochs_no_improve >= args.patience:
                    logging.info(f"[Fold {fold_i}] Early stopping at epoch {epoch}.")
                    break

        # Load best and evaluate test
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

        test_loss, test_probs, test_targs, test_metas = eval_epoch(
            model, test_loader, device, criterion
        )
        wm = window_metrics(np.array(test_targs), np.array(test_probs))
        keys, file_probs, file_targs = aggregate_by_file(
            test_metas, test_probs, test_targs
        )
        fm = window_metrics(file_targs, file_probs)

        logging.info(
            f"[Fold {fold_i}] WINDOW metrics: acc={wm['acc']:.3f} f1={wm['f1']:.3f} auc={wm['roc_auc']:.3f} mcc={wm['mcc']:.3f}"
        )
        logging.info(
            f"[Fold {fold_i}] FILE   metrics: acc={fm['acc']:.3f} f1={fm['f1']:.3f} auc={fm['roc_auc']:.3f} mcc={fm['mcc']:.3f}"
        )

        # save fold rows (file-level)
        for (subj, cond, fp), p, t in zip(keys, file_probs, file_targs):
            all_fold_rows.append(
                {
                    "fold": fold_i,
                    "test_subject": test_subj,
                    "eval_unit": "file",
                    "subject": subj,
                    "condition": cond,
                    "file_path": fp,
                    "prob": float(p),
                    "target": int(t),
                }
            )
        # also save a window-level sample summary row
        all_fold_rows.append(
            {
                "fold": fold_i,
                "test_subject": test_subj,
                "eval_unit": "window_summary",
                "subject": test_subj,
                "condition": -1,
                "file_path": "",
                "prob": float(np.mean(test_probs)),
                "target": int(round(float(np.mean(test_targs)))),
                "win_acc": wm["acc"],
                "win_f1": wm["f1"],
                "win_auc": wm["roc_auc"],
                "win_mcc": wm["mcc"],
                "file_acc": fm["acc"],
                "file_f1": fm["f1"],
                "file_auc": fm["roc_auc"],
                "file_mcc": fm["mcc"],
            }
        )

    # Save CSVs
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    per_item_csv = os.path.join(out_dir, "cnn_lstm_results.csv")
    df = pd.DataFrame(all_fold_rows)
    df.to_csv(per_item_csv, index=False)
    logging.info(f"Saved fold results to {per_item_csv}")

    # Build tidy per-fold/per-model summary (use file-level rows)
    file_rows = df[df["eval_unit"] == "file"].copy()
    # subject-level metrics: threshold file prob at 0.5 then compute metrics per test subject?
    # Simpler: compute global metrics over all files (each (subject,condition) counts once)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        matthews_corrcoef,
    )

    y_true = file_rows["target"].values
    y_prob = file_rows["prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    # handle AUC edge case
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    summary = pd.DataFrame(
        [
            {
                "acc": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": auc,
                "mcc": (
                    matthews_corrcoef(y_true, y_pred)
                    if len(np.unique(y_pred)) > 1
                    else 0.0
                ),
            }
        ]
    )
    summary_csv = os.path.join(out_dir, "cnn_lstm_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved summary to {summary_csv}")
    logging.info("Summary:\n" + str(summary))


if __name__ == "__main__":
    main()
