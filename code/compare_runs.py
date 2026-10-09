"""Rank runs by validation accuracy on their training labels (the only selection signal).

    python code/compare_runs.py                 # all noisy-label runs
    python code/compare_runs.py --label synthetic

val_acc_clean is printed for the record only; it is never used to pick a run.
"""
import argparse
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "outputs", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="noisy")
    ap.add_argument("--prefix", default="", help="only runs whose name starts with this")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, a.prefix + "*"))):
        if not os.path.isfile(os.path.join(d, "log.csv")) or not os.path.isfile(os.path.join(d, "config.json")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("label") != a.label or cfg.get("cv_folds"):
            continue
        log = pd.read_csv(os.path.join(d, "log.csv"))
        b = log.loc[log.val_acc_given.idxmax()]
        rows.append(dict(run=os.path.basename(d), best_epoch=int(b.epoch), val_given=round(b.val_acc_given, 4),
                         val_given_final=round(log.val_acc_given.iloc[-1], 4), epochs=len(log),
                         clean_at_best=round(b.val_acc_clean, 4)))
    df = pd.DataFrame(rows).sort_values("val_given", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
