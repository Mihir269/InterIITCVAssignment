"""Assemble the submission folder from the run outputs, fill results.json, run check_format.py and zip it.

    python code/make_submission.py --roll_no <rollno> --name "<name>"

Reads outputs/runs/<run>/{log.csv,pred_*.csv}, outputs/task2/summary.json and outputs/task3/chosen.json.
Every validation number reported here is measured with NOISY validation labels (Tasks 1-4) or
SYNTHETIC validation labels (Task 5), as results.json asks.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.path.join(CV, "outputs", "runs")
SUBMIT_CODE = ["model.py", "train.py", "ensemble.py", "analyze_noise.py", "task2_analyze.py",
               "task3_scores.py", "task4_targets.py", "feature_scores.py", "handcrafted_scores.py",
               "noise_difficulty.py", "make_submission.py"]

# Default Task 4 methods: (results.json name, run dir, prediction file name in predictions/).
# Override with --t4 name:run_dir (file = t4_<name>.csv).
T4 = [
    ("elr_mixup", "t4_elr_mix1_ens", "t4_elr_mixup.csv"),                    # ELR (lambda 3, beta .7) + mixup 1, 2 nets avg
    ("cv_relabel_mixup", "t4_cvrelabel_mix2_ens", "t4_cv_relabel_mixup.csv"),  # beta .2 targets, mixup 2, 2 nets avg
    ("gce_mixup", "t4_gce_mixup_ens", "t4_gce_mixup.csv"),                    # GCE q .7, mixup 1, 2 nets averaged
]
T3_METHODS = {
    "insample_aum": "Area under the margin (AUM) from the Task 1 noisy CE run: for each training image, the margin "
                    "max_{k != y} p_k - p_y (y = noisy label, eval mode, no augmentation) averaged over epochs 1..43 "
                    "(the early-stop epoch chosen with noisy validation accuracy, i.e. before memorization). "
                    "Compared against 5-fold cross-validated held-out loss/margin; AUM ranked best.",
    "cv_loss_avg": "5-fold cross-validation on noisy labels (5 runs). Each image is scored by the fold model that "
                   "never trained on it: mean held-out cross-entropy of its noisy label over epochs 1..e*, where e* "
                   "is that fold's best noisy-validation epoch.",
    "cv_loss": "5-fold cross-validation on noisy labels: held-out cross-entropy of the noisy label at each fold's "
               "best noisy-validation epoch.",
    "cv_margin": "5-fold cross-validation on noisy labels: held-out margin (max other-class prob - prob of noisy "
                 "label) at each fold's best noisy-validation epoch.",
    "rank_mix": "Average rank of (a) 5-fold cross-validated held-out loss and (b) in-sample area-under-the-margin "
                "of the Task 1 noisy CE run up to its early-stop epoch.",
}


def log(run):
    return pd.read_csv(os.path.join(RUNS, run, "log.csv"))


def best_row(run):
    d = log(run)
    return d.loc[d.val_acc_given.idxmax()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_no", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(CV, "submission"))
    ap.add_argument("--t5_best_run", required=True, help="run dir of the best Task 4 method trained on synthetic labels")
    ap.add_argument("--t4", nargs="*", default=None, help="name:run_dir pairs for the Task 4 methods (1-3)")
    a = ap.parse_args()
    t4 = T4 if not a.t4 else [(x.split(":")[0], x.split(":")[1], f"t4_{x.split(':')[0]}.csv") for x in a.t4]

    sub = a.out
    pred = os.path.join(sub, "predictions")
    shutil.rmtree(sub, ignore_errors=True)
    os.makedirs(pred)

    # ---- predictions
    shutil.copy(os.path.join(RUNS, "t1_clean", "pred_final.csv"), os.path.join(pred, "t1_clean_ref.csv"))
    shutil.copy(os.path.join(RUNS, "t1_noisy", "pred_final.csv"), os.path.join(pred, "t1_noisy_ce.csv"))
    shutil.copy(os.path.join(RUNS, "t1_noisy", "dynamics.csv"), os.path.join(pred, "t2_dynamics.csv"))
    t2 = json.load(open(os.path.join(CV, "outputs", "task2", "summary.json")))
    es = int(t2["early_stop_epoch"])
    tl = np.load(os.path.join(RUNS, "t1_noisy", "test_logits.npy"), mmap_mode="r")
    pd.DataFrame({"id": np.arange(tl.shape[1]), "pred": tl[es - 1].argmax(1)}).to_csv(
        os.path.join(pred, "t2_early_stop.csv"), index=False)
    shutil.copy(os.path.join(CV, "predictions", "t3_noise_scores.csv"), os.path.join(pred, "t3_noise_scores.csv"))

    methods = []
    for name, run, fname in t4:
        if not os.path.exists(os.path.join(RUNS, run, "pred_best.csv")):
            print(f"skip Task 4 method {name}: {run} not found")
            continue
        shutil.copy(os.path.join(RUNS, run, "pred_best.csv"), os.path.join(pred, fname))
        methods.append(dict(name=name, file=fname, val_acc_noisy_labels=round(float(best_row(run).val_acc_given), 4)))
    best = max(methods, key=lambda m: m["val_acc_noisy_labels"])
    best_run = {n: r for n, r, _ in t4}[best["name"]]

    # Task 5: baseline = same CE recipe as Task 1 (final epoch, like t1_noisy_ce); best method = its best
    # synthetic-validation epoch, like its Task 4 counterpart.
    shutil.copy(os.path.join(RUNS, "t5_baseline_synth", "pred_final.csv"), os.path.join(pred, "t5_baseline_synth.csv"))
    t5b = a.t5_best_run
    shutil.copy(os.path.join(RUNS, t5b, "pred_best.csv"), os.path.join(pred, "t5_best_synth.csv"))

    # ---- results.json
    labels = pd.read_csv(os.path.join(CV, "data", "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(CV, "data", "train_idx.txt"), dtype=np.int64)
    noise_rate = float((labels.noisy_label.values[tr] != labels.clean_label.values[tr]).mean())
    chosen = json.load(open(os.path.join(CV, "outputs", "task3", "chosen.json")))["signal"]
    results = {
        "name": a.name,
        "roll_no": a.roll_no,
        "seed": 0,
        "training_setup": {
            "optimizer": "SGD, momentum 0.9 (Nesterov), weight decay 5e-4",
            "learning_rate_schedule": "OneCycle per batch: cosine warm-up 0.01->0.1 over the first 15% of steps, "
                                      "then cosine decay to 1e-5 at epoch 60",
            "batch_size": 128,
            "epochs": 60,
            "augmentation": "random crop 32x32 with 4-pixel zero padding + random horizontal flip "
                            "(mixed-precision fp16 training)",
            "gpu_used": "NVIDIA Tesla T4 (Kaggle), one GPU per run",
        },
        "task1": {
            "noise_rate": round(noise_rate, 4),
            "val_acc_noisy_labels__clean_ref_model": round(float(log("t1_clean").val_acc_noisy.iloc[-1]), 4),
            "val_acc_noisy_labels__noisy_ce_model": round(float(log("t1_noisy").val_acc_noisy.iloc[-1]), 4),
        },
        "task2": {
            "memorization_start_epoch": int(t2["memorization_start_epoch"]),
            "early_stop_epoch": es,
            "val_acc_noisy_labels_at_early_stop": round(float(t2["val_acc_noisy_at_early_stop"]), 4),
        },
        "task3": {"method": T3_METHODS.get(chosen, chosen)},
        "task4": {"methods": methods, "best_method": best["name"]},
        "task5": {
            "val_acc_synthetic_labels__baseline": round(float(log("t5_baseline_synth").val_acc_synthetic.iloc[-1]), 4),
            "val_acc_synthetic_labels__best_method": round(float(best_row(t5b).val_acc_synthetic), 4),
        },
    }
    with open(os.path.join(sub, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nbest Task 4 method: {best['name']} (run {best_run})")

    # ---- code, report
    # only the code needed to reproduce the submitted files and the numbers quoted in report.md
    os.makedirs(os.path.join(sub, "code"))
    for f in SUBMIT_CODE:
        shutil.copy(os.path.join(CV, "code", f), os.path.join(sub, "code", f))
    shutil.copy(os.path.join(CV, "code", "README_submission.md"), os.path.join(sub, "code", "README.md"))
    for f in ["report.md"]:
        if os.path.exists(os.path.join(CV, f)):
            shutil.copy(os.path.join(CV, f), os.path.join(sub, f))

    # ---- format check + zip
    subprocess.run([sys.executable, os.path.join(CV, "check_format.py"), sub], cwd=CV)
    zpath = os.path.join(CV, f"{a.roll_no}_cv_bootcamp")
    shutil.make_archive(zpath, "zip", sub)
    subprocess.run([sys.executable, os.path.join(CV, "check_format.py"), zpath + ".zip"], cwd=CV)
    print(f"\nwrote {zpath}.zip")


if __name__ == "__main__":
    main()
