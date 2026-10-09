"""Estimate the noise transition matrix T[i, j] = P(given label = j | true class = i) WITHOUT clean labels.

Anchor-point estimator (Patrini et al. 2017, Xia et al. 2019) on the held-out probabilities of the 5-fold
cross-validation models (each image scored by the model that never trained on it):
for every class i, take the training images with the highest held-out p(i) (top `--anchor_frac` of all
images by p(i)); treat them as truly class i and read off the distribution of their given labels.

Clean labels are used only afterwards to MEASURE how close the estimate is (allowed measurement).

    python code/estimate_T.py                                         # real noise   -> outputs/aux/T_est.npy
    python code/estimate_T.py --label synthetic --prefix t5_cv --suffix _synth
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
COLS = {"noisy": "noisy_label", "synthetic": "synthetic_label"}


def heldout_probs(fold_dirs, n):
    P = np.zeros((n, 10), np.float32)
    for d in fold_dirs:
        log = pd.read_csv(os.path.join(d, "log.csv"))
        e = int(log.loc[log.val_acc_given.idxmax(), "epoch"])
        idx = np.load(os.path.join(d, "holdout_index.npy"))
        z = np.load(os.path.join(d, "holdout_logits.npy"))[e - 1].astype(np.float32)
        z = np.exp(z - z.max(1, keepdims=True))
        P[idx] = z / z.sum(1, keepdims=True)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=list(COLS), default="noisy")
    ap.add_argument("--prefix", default="t3_cv")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--anchor_frac", type=float, default=0.01, help="share of all images used as anchors per class")
    ap.add_argument("--data_dir", default=os.path.join(CV, "data"))
    a = ap.parse_args()

    labels = pd.read_csv(os.path.join(a.data_dir, "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(a.data_dir, "train_idx.txt"), dtype=np.int64)
    y = labels[COLS[a.label]].values[tr]
    folds = sorted(d for d in glob.glob(os.path.join(CV, "outputs", "runs", a.prefix + "*")) if os.path.isdir(d))
    P = heldout_probs(folds, len(labels))[tr]

    k = max(10, int(a.anchor_frac * len(tr)))
    T = np.zeros((10, 10))
    for i in range(10):
        anchors = np.argsort(-P[:, i])[:k]
        T[i] = np.bincount(y[anchors], minlength=10) / k
    os.makedirs(os.path.join(CV, "outputs", "aux"), exist_ok=True)
    np.save(os.path.join(CV, "outputs", "aux", f"T_est{a.suffix}.npy"), T.astype(np.float32))

    # ---- measurement only: compare with the true matrix
    c = labels.clean_label.values[tr]
    T_true = np.zeros((10, 10))
    np.add.at(T_true, (c, y), 1)
    T_true /= T_true.sum(1, keepdims=True)
    names = open(os.path.join(a.data_dir, "classes.txt")).read().split()
    np.set_printoptions(precision=2, suppress=True)
    print(f"anchors per class: {k}")
    print("estimated T (rows = true class, cols = given label):")
    print(pd.DataFrame(T, index=names, columns=[n[:5] for n in names]).round(2).to_string())
    print("\n[measurement] true T:")
    print(pd.DataFrame(T_true, index=names, columns=[n[:5] for n in names]).round(2).to_string())
    print(f"\n[measurement] mean abs error {np.abs(T - T_true).mean():.4f}; "
          f"diagonal est {np.diag(T).mean():.3f} vs true {np.diag(T_true).mean():.3f}")


if __name__ == "__main__":
    main()
