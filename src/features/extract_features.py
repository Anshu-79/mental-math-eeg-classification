"""Extract Welch bandpower, Hjorth parameters, and spectral entropy from windows.
Usage: python -m src.features.extract_features --indir data/derivatives/raw_windows --out features_windows.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import welch
from scipy.stats import entropy


def bandpower(data, sfreq, band):
    f, Pxx = welch(data, fs=sfreq, nperseg=sfreq * 2)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])


def hjorth_params(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_zero = np.var(x)
    var_d1 = np.var(dx)
    var_d2 = np.var(ddx)
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = (
        np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 and mobility > 0 else 0
    )
    return activity, mobility, complexity


def spectral_entropy(x, sfreq):
    f, Pxx = welch(x, fs=sfreq, nperseg=sfreq * 2)
    Pxx = Pxx / np.sum(Pxx)
    return entropy(Pxx)


def extract_features_from_window(win, sfreq, bands):
    feats = {}
    for name, (low, high) in bands.items():
        bp = bandpower(win, sfreq, (low, high))
        feats[f"band_{name}"] = np.log(bp + 1e-8)
    feats["ratio_theta_beta"] = feats["band_theta"] / (feats["band_beta"] + 1e-8)
    feats["ratio_alpha_thetabeta"] = feats["band_alpha"] / (
        feats["band_theta"] + feats["band_beta"] + 1e-8
    )
    act, mob, comp = hjorth_params(win)
    feats["hjorth_activity"] = act
    feats["hjorth_mobility"] = mob
    feats["hjorth_complexity"] = comp
    feats["spectral_entropy"] = spectral_entropy(win, sfreq)
    return feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()


bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


rows = []
for fpath in sorted(glob(os.path.join(args.indir, "*.npz"))):
    dat = np.load(fpath, allow_pickle=True)
    windows, sfreq, chs = dat["windows"], float(dat["sfreq"]), dat["channels"]
    subj_cond = os.path.basename(fpath).replace(".npz", "")
    subj, cond = subj_cond.split("_", 1)
    for i, win in enumerate(windows):
        # average features across channels
        ch_feats = [
            extract_features_from_window(win[ch], sfreq, bands)
            for ch in range(win.shape[0])
        ]
        avg_feats = {k: np.mean([cf[k] for cf in ch_feats]) for k in ch_feats[0].keys()}
        avg_feats.update({"subject": subj, "condition": cond, "window": i})
        rows.append(avg_feats)


df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"Wrote {len(df)} rows to {args.out}")
