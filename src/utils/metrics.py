import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


def window_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # ROC-AUC needs both classes; handle edge cases
    auc = np.nan
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    return {"acc": acc, "f1": f1, "roc_auc": auc, "mcc": mcc}


def aggregate_by_file(meta_list, y_prob_list, y_true_list):
    """
    Aggregate window-level probabilities to file-level (subject-condition) by mean prob.
    Returns arrays aligned per unique (subject, condition, file_path).
    """
    # group keys
    keys = [(m["subject"], m["condition"], m["file_path"]) for m in meta_list]
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, []).append(i)
    agg_probs, agg_targets, agg_keys = [], [], []
    for k, idxs in uniq.items():
        probs = np.array([y_prob_list[i] for i in idxs])
        targs = np.array([y_true_list[i] for i in idxs])
        agg_probs.append(float(np.mean(probs)))
        # all windows from a file share same label; take majority to be safe
        agg_targets.append(int(round(float(np.mean(targs)))))
        agg_keys.append(k)
    return agg_keys, np.array(agg_probs), np.array(agg_targets)
