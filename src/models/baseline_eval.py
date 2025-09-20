import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import logging
import sys
import os

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ----------------------------
# Baseline evaluation
# ----------------------------
def evaluate_baselines(features_csv, out_csv):
    logging.info(f"Loading features from {features_csv}")
    df = pd.read_csv(features_csv)
    logging.info(
        f"Loaded dataframe with shape {df.shape} and columns: {list(df.columns)}"
    )

    # Features, labels, groups
    X = df.drop(columns=["subject", "condition", "window"])
    y = df["condition"].astype(int).values
    groups = df["subject"].values

    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Unique subjects: {len(np.unique(groups))}")
    logging.info(f"Label distribution: {np.bincount(y)} (labels are {np.unique(y)})")

    logo = LeaveOneGroupOut()
    results = []

    # ----------------------------
    # LOSO cross-validation
    # ----------------------------
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        fold += 1
        test_subject = groups[test_idx][0]
        logging.info(f"\n=== Fold {fold}: Test subject {test_subject} ===")
        logging.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "LogReg": LogisticRegression(max_iter=500, class_weight="balanced"),
            "RF": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
        }

        for name, model in models.items():
            logging.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = None
                roc = np.nan

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            mcc = matthews_corrcoef(y_test, y_pred)

            logging.info(
                f"{name} → Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc:.3f}, MCC: {mcc:.3f}"
            )

            results.append(
                {
                    "subject": test_subject,
                    "model": name,
                    "acc": acc,
                    "f1": f1,
                    "roc_auc": roc,
                    "mcc": mcc,
                }
            )

    results_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Save per-fold results
    results_df.to_csv(out_csv, index=False)
    logging.info(f"\nSaved all results to {out_csv}")

    # Per-model averages
    logging.info(
        f"Overall mean results:\n{results_df.groupby('model').mean(numeric_only=True)}"
    )

    # ----------------------------
    # Save summary metrics
    # ----------------------------
    numeric_cols = results_df.select_dtypes(include="number")
    summary = numeric_cols.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]

    summary_csv = os.path.join(os.path.dirname(out_csv), "baseline_summary.csv")
    summary.to_csv(summary_csv, index=False)

    logging.info(f"\nSaved summary metrics to {summary_csv}")
    logging.info("\nFinal averaged metrics:\n" + str(summary))


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=str, required=True, help="Path to features_windows.csv"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/baseline_results.csv",
        help="Output CSV for fold-wise results",
    )
    args = parser.parse_args()

    evaluate_baselines(args.features, args.out)
