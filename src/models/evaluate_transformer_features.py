#!/usr/bin/env python3
"""
Evaluate feature-based transformer checkpoints (fold*_best.pt) on the feature dataset.

Usage:
    python evaluate_transformer_features.py --features path/to/features.csv --ckpt_dir reports/checkpoints_transformer --out_dir reports/eval_features
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure project root is importable like training did
sys.path.append(os.path.abspath("."))

# reuse metric helpers you used during training
from src.utils.metrics import window_metrics, aggregate_by_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

# ----------------------------
# Logging (ASCII safe)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ----------------------------
# Recreate Transformer used in training (must match names/shape)
# ----------------------------

class EEGTransformer(torch.nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        # Same as training: single Linear layer + Transformer + output
        self.input_fc = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch, input_dim]
        x = self.input_fc(x.unsqueeze(1))  # [batch, 1, d_model]
        x = self.encoder(x)  # [batch, 1, d_model]
        x = x.mean(dim=1)  # [batch, d_model]
        logits = self.fc(x)  # [batch, 1]
        return logits.squeeze(-1)


# ----------------------------
# eval helpers (matching training)
# ----------------------------
@torch.no_grad()
def eval_epoch_features(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    probs, targets, metas = [], [], []
    for x, y, subj, win in loader:
        x = x.to(device)
        # labels in your training code were 1/2 -> you did (y - 1).float()
        # keep raw y here (1/2) and convert to 0/1 when computing loss and metrics
        y_orig = y
        y = (y - 1).float().to(device)  # 1->0, 2->1
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.tolist())
        targets.extend(y.cpu().numpy().astype(int).tolist())
        metas.extend(list(zip(subj, win)))
    avg_loss = (
        total_loss / len(loader.dataset)
        if (criterion is not None and len(loader.dataset) > 0)
        else float("nan")
    )
    return avg_loss, np.array(probs), np.array(targets), metas


# ----------------------------
# Main evaluation routine
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        help="Path to feature CSV used to create EEGFeatureDataset",
    )
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Directory containing fold{fold}_best.pt checkpoints",
    )
    parser.add_argument(
        "--out_dir", default="reports/eval_features", help="Where to save results"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load dataset using the original dataset class to replicate splits
    from src.data.feature_dataset import EEGFeatureDataset

    ds = EEGFeatureDataset(args.features)
    logger.info(
        f"Loaded EEGFeatureDataset: {len(ds)} samples from {len(ds.unique_subjects)} subjects"
    )
    unique_subjects = (
        ds.unique_subjects
    )  # training used dataset.unique_subjects in same order

    # find checkpoint files
    ckpt_files = sorted(
        [
            f
            for f in os.listdir(args.ckpt_dir)
            if f.startswith("fold") and f.endswith("_best.pt")
        ]
    )
    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")

    per_fold_rows = []  # file-level rows like training script
    per_window_rows = []  # optional: save per-window probs for downstream analysis

    criterion = nn.BCEWithLogitsLoss()

    for fold_idx, ckpt_name in enumerate(ckpt_files, start=1):
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        # training enumerated folds over dataset.unique_subjects with start=1
        try:
            test_subject = unique_subjects[fold_idx - 1]
        except IndexError:
            logger.warning(
                f"Fold index {fold_idx} out of range of unique_subjects. Skipping {ckpt_name}."
            )
            continue

        logger.info(
            f"=== Evaluating Fold {fold_idx} | Test subject: {test_subject} ==="
        )
        # recreate train/test split exactly like training script
        train_ds, test_ds = ds.split_subject(test_subject)
        # pick val subject from train subjects as training did
        train_subjects = sorted(train_ds.unique_subjects)
        val_subject = train_subjects[-1]  # same heuristic from training
        val_mask = np.array([s == val_subject for s in train_ds.subjects])
        train_mask = np.array([s != val_subject for s in train_ds.subjects])

        # build final train/val/test objects like training (we only need test loader for eval)
        # (reuse EEGFeatureDataset.__new__ trick if needed) - but easiest: use provided split_subject above for test
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # instantiate model with correct input dimension (feature dimension)
        input_dim = train_ds.features.shape[1]  # same as training
        model = EEGTransformer(input_dim=input_dim).to(args.device)

        # load checkpoint
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        # ensure compatibility with names 'fc' in saved state dict
        # training used layer name 'fc'
        # our model defines self.fc so keys should match; load strictly
        try:
            model.load_state_dict(sd)
        except RuntimeError as e:
            # try less strict: maybe saved with "module." prefix or different key container
            # handle "module." keys (DataParallel)
            sd2 = {}
            for k, v in sd.items():
                newk = k.replace("module.", "") if k.startswith("module.") else k
                sd2[newk] = v
            try:
                model.load_state_dict(sd2)
            except Exception as e2:
                logger.error(f"Failed to load state_dict for {ckpt_path}: {e2}")
                logger.info("Attempting flexible load (load matching keys only).")
                model_dict = model.state_dict()
                matched = {
                    k: v
                    for k, v in sd2.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(matched)
                model.load_state_dict(model_dict)
                logger.warning(
                    f"Flexible load used for {ckpt_path}. Matched keys: {len(matched)} / {len(model_dict)}"
                )

        # evaluate test set
        test_loss, test_probs, test_targs, test_metas = eval_epoch_features(
            model, test_loader, args.device, criterion
        )

        # window metrics (on windows, test_targs already 0/1)
        wm = window_metrics(np.array(test_targs), np.array(test_probs))

        # aggregate to file-level using your training utility
        # construct test_metas_file similar to training code: each meta is (subject, condition? , file_path)
        # In EEGFeatureDataset, sample metadata columns are available as test_ds.data or test_ds.subjects / windows indices.
        # test_metas from eval_epoch_features is list(zip(subj, win_idx)) — we will build file entries
        test_metas_file = [
            {"subject": m[0], "condition": test_targs[i], "file_path": str(i)}
            for i, m in enumerate(test_metas)
        ]
        keys, file_probs, file_targs = aggregate_by_file(
            test_metas_file, test_probs, test_targs
        )

        fm = window_metrics(file_targs, file_probs)

        # logging per-fold metrics
        logger.info(
            f"[Fold {fold_idx}] WINDOW metrics: acc={wm['acc']:.3f} f1={wm['f1']:.3f} auc={wm['roc_auc']:.3f} mcc={wm['mcc']:.3f}"
        )
        logger.info(
            f"[Fold {fold_idx}] FILE   metrics: acc={fm['acc']:.3f} f1={fm['f1']:.3f} auc={fm['roc_auc']:.3f} mcc={fm['mcc']:.3f}"
        )

        # append per-file rows (same format as training)
        for (subj, cond, fp), p, t in zip(keys, file_probs, file_targs):
            per_fold_rows.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    "eval_unit": "file",
                    "subject": subj,
                    "condition": cond,
                    "file_path": fp,
                    "prob": float(p),
                    "target": int(t),
                }
            )

        # also append a window-summary row (like training)
        per_fold_rows.append(
            {
                "fold": fold_idx,
                "test_subject": test_subject,
                "eval_unit": "window_summary",
                "subject": test_subject,
                "condition": -1,
                "file_path": "",
                "prob": float(np.mean(test_probs)) if len(test_probs) > 0 else 0.0,
                "target": (
                    int(round(float(np.mean(test_targs)))) if len(test_targs) > 0 else 0
                ),
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

        # optionally save per-window probabilities for later analysis
        for (subj, win_idx), p, t in zip(test_metas, test_probs, test_targs):
            per_window_rows.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    "subject": subj,
                    "window_idx": win_idx,
                    "prob": float(p),
                    "target": int(t),
                }
            )

    # Save outputs
    per_fold_csv = os.path.join(args.out_dir, "transformer_feature_fold_metrics.csv")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False)
    logger.info(f"Saved per-fold results to {per_fold_csv}")

    per_window_csv = os.path.join(args.out_dir, "transformer_feature_per_window.csv")
    pd.DataFrame(per_window_rows).to_csv(per_window_csv, index=False)
    logger.info(f"Saved per-window probabilities to {per_window_csv}")

    # build summary (file-level rows only)
    file_rows = pd.DataFrame(per_fold_rows)
    file_rows = file_rows[file_rows["eval_unit"] == "file"]
    if len(file_rows) > 0:
        y_true = file_rows["target"].astype(int).values
        y_prob = file_rows["prob"].values
        y_pred = (y_prob >= 0.5).astype(int)
        auc = (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        summary = pd.DataFrame(
            [
                {
                    "acc": accuracy_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "roc_auc": auc,
                    "mcc": mcc,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [{"acc": np.nan, "f1": np.nan, "roc_auc": np.nan, "mcc": np.nan}]
        )

    summary_csv = os.path.join(args.out_dir, "transformer_feature_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary to {summary_csv}")
    logger.info("Summary:\n" + str(summary))


if __name__ == "__main__":
    main()
