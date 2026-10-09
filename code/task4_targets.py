"""Task 4 inputs built from 5-fold cross-validation models (training labels only, no clean labels):

  outputs/aux/targets_cvrelabel{suffix}.npy  (50000, 10) soft targets = beta * onehot(label) + (1 - beta) * p_heldout
  outputs/aux/forget_rate{suffix}.txt        estimated noise rate = share of training images whose held-out
                                             prediction disagrees with the label (used by co-teaching)

p_heldout comes from the fold model that never trained on the image, at that fold's best validation epoch
(validation accuracy against the same label column). beta is fixed a priori (0.2): low enough that a label
the held-out model rejects (p_y ~ 0.1) gets only ~0.28 of the target mass, high enough to keep some weight
on labels where the model is unsure.

    python code/task4_targets.py                                           # real noise, folds t3_cv*
    python code/task4_targets.py --label synthetic --prefix t5_cv --suffix _synth   # Task 5
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
        print(f"{os.path.basename(d)}: best validation epoch {e}")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--label", choices=list(COLS), default="noisy")
    ap.add_argument("--prefix", default="t3_cv")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--data_dir", default=os.path.join(CV, "data"))
    ap.add_argument("--runs", default=os.path.join(CV, "outputs", "runs"))
    a = ap.parse_args()

    labels = pd.read_csv(os.path.join(a.data_dir, "labels.csv")).set_index("index").sort_index()
    train_idx = np.loadtxt(os.path.join(a.data_dir, "train_idx.txt"), dtype=np.int64)
    y = labels[COLS[a.label]].values
    folds = sorted(d for d in glob.glob(os.path.join(a.runs, a.prefix + "*")) if os.path.isdir(d))
    assert len(folds) == 5, f"expected 5 fold runs {a.prefix}*, found {len(folds)}"
    P = heldout_probs(folds, len(labels))
    assert np.allclose(P[train_idx].sum(1), 1, atol=1e-3), "missing held-out probs for some training images"

    T = a.beta * np.eye(10, dtype=np.float32)[y] + (1 - a.beta) * P
    T[np.setdiff1d(np.arange(len(T)), train_idx)] = 0  # only training rows are used

    disagree = (P[train_idx].argmax(1) != y[train_idx]).mean()
    out = os.path.join(CV, "outputs", "aux")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"targets_cvrelabel{a.suffix}.npy"), T.astype(np.float16))
    with open(os.path.join(out, f"forget_rate{a.suffix}.txt"), "w") as f:
        f.write(f"{disagree:.4f}\n")
    changed = (T[train_idx].argmax(1) != y[train_idx]).mean()
    print(f"estimated noise rate (held-out disagreement): {disagree:.4f}")
    print(f"targets whose argmax differs from the given label: {changed:.4f}")
    print(f"wrote {out}/targets_cvrelabel{a.suffix}.npy and forget_rate{a.suffix}.txt")


if __name__ == "__main__":
    main()
