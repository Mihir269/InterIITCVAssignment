"""Task 1/2 analysis of the CE runs: memorization point, early-stop epoch, plots and prediction files.

    python code/task2_analyze.py

* early-stop epoch  = epoch with the highest NOISY validation accuracy (the only selection signal used).
* memorization start (measured with clean labels on the training set, which the rules allow for the
  Task 2 curves) = first epoch after which accuracy on the mislabeled subset (i.e. agreement with the
  WRONG label) keeps rising above its early-training floor, reported together with the epoch where
  clean validation accuracy peaks.
"""
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")


def memorization_start(log, n_sigma=3):
    """First epoch from which accuracy on the mislabeled subset (= agreement with the WRONG label) stays
    above its early-training level, mean + n_sigma * std over the first half of training, for good."""
    m = log.train_acc_mislabeled.values
    half = m[: len(m) // 2]
    thr = half.mean() + n_sigma * half.std()
    above = m > thr
    lo = int(np.argmin(m))
    for i in range(len(m)):
        if above[i:].all():
            return int(log.epoch.iloc[i]), int(log.epoch.iloc[lo])
    return None, int(log.epoch.iloc[lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(CV, "outputs", "runs"))
    ap.add_argument("--pred_dir", default=os.path.join(CV, "predictions"))
    args = ap.parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)
    out = os.path.join(CV, "outputs", "task2")
    os.makedirs(out, exist_ok=True)

    noisy_dir = os.path.join(args.runs, "t1_noisy")
    log = pd.read_csv(os.path.join(noisy_dir, "log.csv"))
    es = int(log.loc[log.val_acc_noisy.idxmax(), "epoch"])
    mem, floor = memorization_start(log)
    clean_peak = int(log.loc[log.val_acc_clean.idxmax(), "epoch"])
    r = log.set_index("epoch")

    print("noisy CE run")
    print(f"  early-stop epoch (max noisy val acc): {es}   noisy val {r.val_acc_noisy[es]:.4f}  "
          f"clean val {r.val_acc_clean[es]:.4f}")
    print(f"  final epoch {log.epoch.iloc[-1]}: noisy val {log.val_acc_noisy.iloc[-1]:.4f}  "
          f"clean val {log.val_acc_clean.iloc[-1]:.4f}")
    print(f"  clean val acc peaks at epoch {clean_peak} ({r.val_acc_clean[clean_peak]:.4f})")
    print(f"  train acc on mislabeled subset: floor {r.train_acc_mislabeled[floor]:.4f} at epoch {floor}, "
          f"memorization start (sustained rise) at epoch {mem}")
    print(f"  final train acc: correct subset {log.train_acc_correct.iloc[-1]:.4f}, "
          f"mislabeled subset {log.train_acc_mislabeled.iloc[-1]:.4f}")

    # Task 1 / 2 prediction files
    shutil.copy(os.path.join(noisy_dir, "dynamics.csv"), os.path.join(args.pred_dir, "t2_dynamics.csv"))
    shutil.copy(os.path.join(noisy_dir, "pred_final.csv"), os.path.join(args.pred_dir, "t1_noisy_ce.csv"))
    test_logits = np.load(os.path.join(noisy_dir, "test_logits.npy"), mmap_mode="r")
    pd.DataFrame({"id": np.arange(test_logits.shape[1]), "pred": test_logits[es - 1].argmax(1)}) \
        .to_csv(os.path.join(args.pred_dir, "t2_early_stop.csv"), index=False)
    clean_dir = os.path.join(args.runs, "t1_clean")
    if os.path.exists(os.path.join(clean_dir, "pred_final.csv")):
        shutil.copy(os.path.join(clean_dir, "pred_final.csv"), os.path.join(args.pred_dir, "t1_clean_ref.csv"))
        cl = pd.read_csv(os.path.join(clean_dir, "log.csv"))
        print(f"\nclean reference run, final epoch: clean val {cl.val_acc_clean.iloc[-1]:.4f}  "
              f"noisy val {cl.val_acc_noisy.iloc[-1]:.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, lab in [("train_acc_correct", "train acc, correctly-labeled subset"),
                   ("train_acc_mislabeled", "train acc, mislabeled subset (fits WRONG label)"),
                   ("val_acc_noisy", "val acc, noisy labels"), ("val_acc_clean", "val acc, clean labels")]:
        ax.plot(log.epoch, log[c], label=lab)
    ax.axvline(es, color="k", ls="--", lw=1, label=f"early stop (max noisy val) = {es}")
    if mem:
        ax.axvline(mem, color="r", ls=":", lw=1.2, label=f"memorization start = {mem}")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("ResNet-18, cross-entropy on CIFAR-10N worst labels")
    fig.tight_layout(); fig.savefig(os.path.join(out, "dynamics.png"), dpi=130)
    pd.Series(dict(early_stop_epoch=es, memorization_start_epoch=mem, clean_val_peak_epoch=clean_peak,
                   val_acc_noisy_at_early_stop=r.val_acc_noisy[es])).to_json(os.path.join(out, "summary.json"))
    print(f"\nsaved {out}/dynamics.png and predictions/t1_*, t2_*")


if __name__ == "__main__":
    main()
