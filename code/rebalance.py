"""Post-hoc class rebalancing of a trained (2-network) method – no retraining, no labels.

The networks over-predict classes the annotators over-used (e.g. "dog" over "cat"). We measure each class's
share of the networks' own predictions on the 45k TRAINING images (no labels involved), assume the true classes
are balanced (CIFAR-10 has equal classes), and shift the log-probabilities by  -tau * log(share / 0.1).
tau is chosen from a fixed grid by validation accuracy against the training label column (noisy for Task 4,
synthetic for Task 5), exactly like every other selection in this project.

    python code/rebalance.py --ens t4_elr_mix1_ens --members s_elr_mix1 s_elr_mix1_s1 --name t4_elr_mix1_bal

Writes outputs/runs/<name>/ with pred_best.csv, log.csv (one row: the selected setting), summary.json, grid.csv.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
RUNS = os.path.join(CV, "outputs", "runs")
COLS = {"noisy": "noisy_label", "synthetic": "synthetic_label", "clean": "clean_label"}
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def log_softmax(z):
    z = np.asarray(z, np.float64)
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", required=True, help="ensemble run (val/test logits per epoch)")
    ap.add_argument("--members", nargs="+", required=True, help="member runs with train_logits.npy")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    label = json.load(open(os.path.join(RUNS, a.ens, "config.json")))["label"]
    lg = pd.read_csv(os.path.join(RUNS, a.ens, "log.csv"))
    e = int(lg.loc[lg.val_acc_given.idxmax(), "epoch"])          # the method's selected epoch
    P = sum(np.exp(log_softmax(np.load(os.path.join(RUNS, m, "train_logits.npy"), mmap_mode="r")[e - 1]))
            for m in a.members) / len(a.members)
    share = P.mean(0)                                             # predicted class shares on training images
    lv = log_softmax(np.load(os.path.join(RUNS, a.ens, "val_logits.npy"))[e - 1])
    lt = log_softmax(np.load(os.path.join(RUNS, a.ens, "test_logits.npy"))[e - 1])

    L = pd.read_csv(os.path.join(CV, "data", "labels.csv")).set_index("index").sort_index()
    va = np.loadtxt(os.path.join(CV, "data", "val_idx.txt"), dtype=np.int64)
    y = {k: L.loc[va, c].values for k, c in COLS.items()}

    rows = []
    for tau in GRID:
        p = (lv - tau * np.log(share / 0.1)).argmax(1)
        rows.append(dict(tau=tau, val_acc_given=(p == y[label]).mean(), val_acc_noisy=(p == y["noisy"]).mean(),
                         val_acc_synthetic=(p == y["synthetic"]).mean(),
                         val_acc_clean=(p == y["clean"]).mean()))          # clean: measurement only
    g = pd.DataFrame(rows)
    best = g.loc[g.val_acc_given.idxmax()]                         # ties -> smallest tau (first in the grid)
    tau = float(best.tau)
    pt = (lt - tau * np.log(share / 0.1)).argmax(1)

    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    g.to_csv(os.path.join(out, "grid.csv"), index=False)
    pd.DataFrame({"id": np.arange(len(pt)), "pred": pt}).to_csv(os.path.join(out, "pred_best.csv"), index=False)
    pd.DataFrame([dict(epoch=e, **{k: float(best[k]) for k in ["val_acc_given", "val_acc_noisy", "val_acc_synthetic",
                                                               "val_acc_clean"]})]).to_csv(os.path.join(out, "log.csv"), index=False)
    json.dump(dict(label=label, rebalanced_from=a.ens, members=a.members, epoch=e, tau=tau,
                   predicted_share_train=[round(float(x), 4) for x in share]),
              open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump(dict(best_epoch=e, tau=tau, val_acc_given_best=float(best.val_acc_given),
                   val_acc_clean_at_best=float(best.val_acc_clean)), open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(g.round(4).to_string(index=False))
    print(f"selected tau={tau} | test predictions per class: {np.bincount(pt, minlength=10).tolist()}")


if __name__ == "__main__":
    main()
