"""Read EDF files and emit numpy array + metadata.
Usage: python -m src.io.read_edf --datadir data/edf_raw --out data/derivatives/metadata.json
"""

import argparse
import json
import os
from glob import glob
import numpy as np
import pyedflib


def read_edf(path):
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    ch_labels = f.getSignalLabels()
    sfreq = f.getSampleFrequencies()[0]
    sigs = np.vstack([f.readSignal(i) for i in range(n)])  # shape: channels x samples
    f._close()
    del f
    return sigs, float(sfreq), ch_labels


def scan_datadir(datadir):
    records = []
    for fpath in sorted(glob(os.path.join(datadir, "**", "*.edf"), recursive=True)):
        try:
            sigs, sfreq, chs = read_edf(fpath)
            duration = sigs.shape[1] / sfreq
            records.append(
                {
                    "path": fpath,
                    "n_channels": sigs.shape[0],
                    "sfreq": sfreq,
                    "duration_sec": duration,
                    "channels": chs,
                }
            )
        except Exception as e:
            records.append({"path": fpath, "error": str(e)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    recs = scan_datadir(args.datadir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(recs, fh, indent=2)
    print(f"Scanned {len(recs)} files. Wrote metadata to {args.out}")
