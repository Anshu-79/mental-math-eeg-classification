import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import os

# Path to your existing CSV
in_csv = os.getcwd() + "/reports/transformer_results.csv/transformer_results.csv"
out_csv = os.getcwd() + "/reports/transformer_results.csv/transformer_summary.csv"

# Load data
df = pd.read_csv(in_csv)

# Keep only file-level rows
file_rows = df[df["eval_unit"] == "file"].copy()

# Ensure targets are integers (0/1)
y_true = file_rows["target"].astype(int).values
y_prob = file_rows["prob"].values
y_pred = (y_prob >= 0.5).astype(int)

# Handle edge cases
auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
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

# Save summary
summary.to_csv(out_csv, index=False)
print(f"Saved summary to {out_csv}")
print("Summary:\n", summary)
