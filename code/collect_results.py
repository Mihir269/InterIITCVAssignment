"""Collect a small, committable snapshot of all results into results/ (the full outputs/ folder is ~6.5 GB of
logits and weights and is not committed).

    python code/collect_results.py --dest results

Writes:
  results/runs/<run>/{config.json,log.csv,summary.json}   every training run (per-epoch logs)
  results/tables/runs_noisy.csv, runs_synthetic.csv         best epoch by validation on the training labels
  results/tables/real_vs_synthetic.csv                      the submitted methods on both label sets
  results/tables/task3_signals.csv                          Task 3 signal comparison
  results/tables/confusion_*.csv                            noisy-vs-clean confusion matrices
  results/figures/*.png                                     noise matrices, memorization curves

val_acc_clean columns are measurements only; every selection uses val_acc_given.
"""
import argparse
import glob
import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(CV, "outputs")


def best(log):
    b = log.loc[log.val_acc_given.idxmax()]
    return int(b.epoch), float(b.val_acc_given), float(b.val_acc_clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(CV, "results"))
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    for sub in ["runs", "tables", "figures"]:
        os.makedirs(os.path.join(dest, sub), exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(OUT, "runs", "*"))):
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "log.csv")):
            continue
        name = os.path.basename(d)
        o = os.path.join(dest, "runs", name)
        os.makedirs(o, exist_ok=True)
        for f in ["config.json", "log.csv", "summary.json"]:
            if os.path.isfile(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), os.path.join(o, f))
        cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.isfile(os.path.join(d, "config.json")) else {}
        log = pd.read_csv(os.path.join(d, "log.csv"))
        e, g, c = best(log)
        rows.append(dict(run=name, label=cfg.get("label"), cv_fold=cfg.get("cv_fold") if cfg.get("cv_folds") else "",
                         epochs=len(log), best_epoch=e, val_acc_given_best=round(g, 4),
                         val_acc_given_final=round(float(log.val_acc_given.iloc[-1]), 4),
                         val_acc_clean_at_best=round(c, 4),
                         val_acc_clean_final=round(float(log.val_acc_clean.iloc[-1]), 4)))
    df = pd.DataFrame(rows)
    for lab in ["noisy", "synthetic", "clean"]:
        sub = df[df.label == lab].sort_values("val_acc_given_best", ascending=False)
        sub.to_csv(os.path.join(dest, "tables", f"runs_{lab}.csv"), index=False)

    pairs = {"ce_baseline": ("t1_noisy", "t5_baseline_synth"),
             "elr_mixup": ("t4_elr_mix1_ens", "t5_elr_mix1_synth_ens"),
             "cv_relabel_mixup": ("t4_cvrelabel_mix2_ens", "t5_cvrelabel_mix2_synth_ens"),
             "gce_mixup": ("t4_gce_mixup_ens", "t5_gce_mixup_synth_ens")}
    rv = []
    for m, (r, s) in pairs.items():
        lr, ls = (pd.read_csv(os.path.join(OUT, "runs", x, "log.csv")) for x in (r, s))
        (_, gr, cr), (_, gs, cs) = best(lr), best(ls)
        rv.append(dict(method=m, real_val_noisy=round(gr, 4), real_val_clean=round(cr, 4),
                       synth_val_synthetic=round(gs, 4), synth_val_clean=round(cs, 4)))
    pd.DataFrame(rv).to_csv(os.path.join(dest, "tables", "real_vs_synthetic.csv"), index=False)

    for src, dst in [("task3/signal_comparison.csv", "tables/task3_signals.csv"),
                     ("analysis/confusion_train_human.csv", "tables/confusion_train_human.csv"),
                     ("analysis/confusion_train_synthetic.csv", "tables/confusion_train_synthetic.csv"),
                     ("analysis/confusion_val_human.csv", "tables/confusion_val_human.csv"),
                     ("task2/summary.json", "tables/task2_summary.json"),
                     ("analysis/confusion_train.png", "figures/confusion_human_vs_synthetic.png"),
                     ("task2/dynamics.png", "figures/memorization_dynamics.png")]:
        if os.path.isfile(os.path.join(OUT, src)):
            shutil.copy(os.path.join(OUT, src), os.path.join(dest, dst))
    print(f"collected {len(df)} runs into {dest}")


if __name__ == "__main__":
    main()
