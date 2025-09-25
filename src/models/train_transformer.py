import os
import sys
import argparse
import logging
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from src.data.feature_dataset import EEGFeatureDataset
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
    return train_subjects[-1]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y, subj, win in loader:
        x = x.to(device)
        y = (y - 1).float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets, all_meta = [], [], []

    for x, y, subj, win in loader:
        x = x.to(device)
        y = (y - 1).float().to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.cpu().numpy().tolist())
        all_meta.extend(list(zip(subj, win)))

    avg_loss = running_loss / len(loader)
    return avg_loss, all_probs, all_targets, all_meta


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints_transformer")
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = EEGFeatureDataset(args.features)
    logging.info(
        f"Loaded dataset: {len(dataset)} samples from {len(dataset.unique_subjects)} subjects"
    )

    all_fold_rows = []

    for fold, test_subject in enumerate(dataset.unique_subjects, start=1):
        train_ds, test_ds = dataset.split_subject(test_subject)
        train_subjects = sorted(train_ds.unique_subjects)
        val_subject = pick_val_subject(train_subjects)

        # Prepare validation set
        val_mask = np.array([s == val_subject for s in train_ds.subjects])
        train_mask = np.array([s != val_subject for s in train_ds.subjects])

        val_ds = EEGFeatureDataset.__new__(EEGFeatureDataset)
        val_ds.data = train_ds.data[val_mask].reset_index(drop=True)
        val_ds.feature_cols = train_ds.feature_cols
        val_ds.features = train_ds.features[val_mask]
        val_ds.labels = train_ds.labels[val_mask]
        val_ds.subjects = train_ds.subjects[val_mask]
        val_ds.windows = train_ds.windows[val_mask]
        val_ds.unique_subjects = sorted(val_ds.subjects)

        train_ds_eff = EEGFeatureDataset.__new__(EEGFeatureDataset)
        train_ds_eff.data = train_ds.data[train_mask].reset_index(drop=True)
        train_ds_eff.feature_cols = train_ds.feature_cols
        train_ds_eff.features = train_ds.features[train_mask]
        train_ds_eff.labels = train_ds.labels[train_mask]
        train_ds_eff.subjects = train_ds.subjects[train_mask]
        train_ds_eff.windows = train_ds.windows[train_mask]
        train_ds_eff.unique_subjects = sorted(train_ds_eff.subjects)

        # DataLoaders
        train_loader = DataLoader(
            train_ds_eff,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Model, optimizer, loss
        device = args.device
        input_dim = train_ds.features.shape[1]
        model = EEGTransformer(input_dim=input_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        logging.info(f"\n=== Fold {fold} | Test subject: {test_subject} ===")
        logging.info(
            f"Train: {len(train_ds_eff)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
        )
        logging.info(f"Using device: {device}")
        
        # Training loop with early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(ckpt_dir, f"fold{fold}_best.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, _, _, _ = eval_epoch(model, val_loader, criterion, device)
            logging.info(
                f"[Fold {fold}] Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({"state_dict": model.state_dict()}, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logging.info(f"[Fold {fold}] Early stopping at epoch {epoch}.")
                    break

        # Load best model & evaluate test
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_probs, test_targs, test_metas = eval_epoch(
            model, test_loader, criterion, device
        )

        wm = window_metrics(np.array(test_targs), np.array(test_probs))

        # File-level metrics
        test_metas_file = [
            {"subject": m[0], "condition": test_targs[i], "file_path": str(i)}
            for i, m in enumerate(test_metas)
        ]
        keys, file_probs, file_targs = aggregate_by_file(
            test_metas_file, test_probs, test_targs
        )
        fm = window_metrics(file_targs, file_probs)

        logging.info(
            f"[Fold {fold}] WINDOW metrics: acc={wm['acc']:.3f} f1={wm['f1']:.3f} auc={wm['roc_auc']:.3f} mcc={wm['mcc']:.3f}"
        )
        logging.info(
            f"[Fold {fold}] FILE   metrics: acc={fm['acc']:.3f} f1={fm['f1']:.3f} auc={fm['roc_auc']:.3f} mcc={fm['mcc']:.3f}"
        )

        # Save fold rows
        for (subj, cond, fp), p, t in zip(keys, file_probs, file_targs):
            all_fold_rows.append(
                {
                    "fold": fold,
                    "test_subject": test_subject,
                    "eval_unit": "file",
                    "subject": subj,
                    "condition": cond,
                    "file_path": fp,
                    "prob": float(p),
                    "target": int(t),
                }
            )

        # Window summary row
        all_fold_rows.append(
            {
                "fold": fold,
                "test_subject": test_subject,
                "eval_unit": "window_summary",
                "subject": test_subject,
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
    out_csv = os.path.join(args.out_dir, "transformer_results.csv")
    pd.DataFrame(all_fold_rows).to_csv(out_csv, index=False)
    logging.info(f"Saved per-fold results to {out_csv}")

    # Summary
    file_rows = pd.DataFrame(all_fold_rows)
    file_rows = file_rows[file_rows["eval_unit"] == "file"]
    # Map targets from 1/2 -> 0/1
    y_true = file_rows["target"].values - 1  # 1->0, 2->1
    y_prob = file_rows["prob"].values
    y_pred = (y_prob >= 0.5).astype(int)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        matthews_corrcoef,
    )

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
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
    summary_csv = os.path.join(args.out_dir, "transformer_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved summary to {summary_csv}")
    logging.info("Summary:\n" + str(summary))


if __name__ == "__main__":
    main()
