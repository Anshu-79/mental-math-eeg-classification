"""Create 2s windows (50% overlap) and save per subject-condition .npz files.
Assumes EDF filenames contain subject id and condition, e.g. sub-01_rest.edf or sub-01_task.edf
"""

import argparse
import os
from glob import glob
import numpy as np
from pathlib import Path
from src.io.read_edf import read_edf


def sliding_windows(data, sfreq, win_sec=2, hop_sec=1):
    win_samps = int(win_sec * sfreq)
    hop_samps = int(hop_sec * sfreq)
    n = data.shape[1]
    starts = list(range(0, n - win_samps + 1, hop_samps))
    windows = np.stack([data[:, s : s + win_samps] for s in starts], axis=0)
    return windows  # shape: n_windows x channels x win_samps


def parse_subj_cond(fname):
    # Very small heuristic: look for sub-XX and *_task/rest
    b = Path(fname).stem
    parts = b.split("_")
    subj = parts[0] if parts else b
    cond = parts[1] if len(parts) > 1 else "unknown"
    return subj, cond


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--outdir", default="data/derivatives/raw_windows")
    parser.add_argument("--win", type=float, default=2.0)
    parser.add_argument("--hop", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for fpath in sorted(
        glob(os.path.join(args.datadir, "**", "*.edf"), recursive=True)
    ):
        try:
            sigs, sfreq, chs = read_edf(fpath)
            subj, cond = parse_subj_cond(fpath)
            wins = sliding_windows(sigs, sfreq, win_sec=args.win, hop_sec=args.hop)
            outname = f"{subj}_{cond}.npz"
            np.savez_compressed(
                os.path.join(args.outdir, outname),
                windows=wins,
                sfreq=sfreq,
                channels=chs,
            )
            print(f"Wrote {outname}: windows={wins.shape}")
        except Exception as e:
            print(f"FAILED {fpath}: {e}")
