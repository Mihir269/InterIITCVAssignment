"""Noise-corrected validation accuracy: estimate a model's accuracy on the TRUE labels from the NOISY
validation labels plus the transition matrix T estimated without clean labels (code/estimate_T.py).

If annotators flip labels according to T[true, given], then for validation predictions
    M[pred, given] = sum_true C[pred, true] * T[true, given]   i.e.  M = C T   ->   C = M T^-1
and the estimated clean accuracy is trace(C). Only noisy labels and T_est are used.

    python code/noise_corrected_val.py            # all noisy-label runs, evaluated at their best noisy-val epoch

The last column (true clean accuracy) is printed as a measurement of how good the estimate is.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")


def est_clean_acc(pred, given, T_inv):
    M = np.zeros((10, 10))
    np.add.at(M, (pred, given), 1)
    M /= len(pred)
    return float(np.trace(M @ T_inv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default=os.path.join(CV, "outputs", "aux", "T_est.npy"))
    ap.add_argument("--label", default="noisy")
    a = ap.parse_args()
    T_inv = np.linalg.inv(np.load(a.T).astype(np.float64))
    labels = pd.read_csv(os.path.join(CV, "data", "labels.csv")).set_index("index").sort_index()
    val_idx = np.loadtxt(os.path.join(CV, "data", "val_idx.txt"), dtype=np.int64)
    col = {"noisy": "noisy_label", "synthetic": "synthetic_label"}[a.label]
    given, clean = labels.loc[val_idx, col].values, labels.loc[val_idx, "clean_label"].values

    rows = []
    for d in sorted(glob.glob(os.path.join(CV, "outputs", "runs", "*"))):
        f = os.path.join(d, "val_logits.npy")
        if not (os.path.isfile(f) and os.path.isfile(os.path.join(d, "config.json"))):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("label") != a.label or cfg.get("cv_folds"):
            continue
        V = np.load(f, mmap_mode="r")
        log = pd.read_csv(os.path.join(d, "log.csv"))
        e_noisy = int(log.loc[log.val_acc_given.idxmax(), "epoch"])
        est = [est_clean_acc(np.asarray(V[k]).argmax(1), given, T_inv) for k in range(V.shape[0])]
        e_est = int(np.argmax(est)) + 1
        p = np.asarray(V[e_noisy - 1]).argmax(1)
        rows.append(dict(run=os.path.basename(d), noisy_val=(p == given).mean(), est_clean=est[e_noisy - 1],
                         best_est_epoch=e_est, est_clean_at_that=est[e_est - 1],
                         true_clean=(p == clean).mean()))  # measurement only
    df = pd.DataFrame(rows).sort_values("est_clean", ascending=False)
    print(df.round(4).to_string(index=False))
    print(f"\n[measurement] Spearman rank correlation with true clean accuracy: "
          f"noisy_val {spearmanr(df.noisy_val, df.true_clean)[0]:.3f}, est_clean {spearmanr(df.est_clean, df.true_clean)[0]:.3f}")


if __name__ == "__main__":
    main()
