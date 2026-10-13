"""Build a Kaggle notebook that embeds the current code/ files and runs a list of training jobs.

    python kaggle/make_notebook.py                 # -> kaggle/cv_task1.ipynb (clean + noisy runs)
    python kaggle/make_notebook.py --zip_data      # also builds kaggle/cv-bootcamp-data.zip for upload

Re-run this after changing anything in code/, then re-import the notebook on Kaggle.
"""
import argparse
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# project root = folder holding data/ and code/ (kaggle/ sits in it, or in code/ inside the submission zip)
_up = os.path.dirname(HERE)
CV = _up if os.path.isdir(os.path.join(_up, "data")) else os.path.dirname(_up)
CODE_FILES = ["model.py", "train.py", "coteach.py", "dividemix.py", "elrplus.py"]

# (name, arguments[, script]) - script defaults to train.py. Jobs are spread over the available GPUs, one job per GPU at a time.
JOBS = {
    "task1": [
        ("t1_clean", "--label clean"),
        ("t1_noisy", "--label noisy"),
    ],
    # quick check of GPU availability and per-epoch speed
    "diag": [("diag_noisy", "--label noisy --epochs 2"), ("diag_clean", "--label clean --epochs 2")],
    # Task 3: 5-fold cross-validation on noisy labels (5 training runs = the Task 3 budget).
    # Each run saves per-epoch logits on its held-out fold; the score epoch is chosen afterwards
    # with noisy validation accuracy only.
    "task3": [(f"t3_cv{k}", f"--label noisy --cv_folds 5 --cv_fold {k} --no_train_logits") for k in range(5)],
    # Task 4: three methods, same recipe and budget as the noisy CE baseline.
    #  gce        robust loss (1 net)
    #  cvrelabel  retrain on targets mixed with held-out predictions of the Task 3 models (1 net)
    #  coteach    co-teaching, forget rate estimated from the Task 3 models (2 nets)
    "task4": [
        ("t4_coteach", "--label noisy --forget_rate {FILE:forget_rate.txt}", "coteach.py"),
        ("t4_gce", "--label noisy --loss gce"),
        ("t4_cvrelabel", "--label noisy --targets_file {AUX}/targets_cvrelabel.npy"),
    ],
    # Task 5 part 1: CE baseline + GCE on synthetic labels, and synthetic-label CV folds (needed to rebuild
    # the relabel targets / forget rate for the other methods on synthetic noise, for the Q5 ranking).
    "task5a": [("t5_baseline_synth", "--label synthetic"), ("t5_gce_synth", "--label synthetic --loss gce")]
    + [(f"t5_cv{k}", f"--label synthetic --cv_folds 5 --cv_fold {k} --no_train_logits") for k in range(5)],
    # Task 5 part 2: the other two methods on synthetic labels, inputs rebuilt from the synthetic CV folds.
    "task5b": [
        ("t5_coteach_synth", "--label synthetic --forget_rate {FILE:forget_rate_synth.txt}", "coteach.py"),
        ("t5_cvrelabel_synth", "--label synthetic --targets_file {AUX}/targets_cvrelabel_synth.npy"),
    ],
    # Improvement round 1: use the full 2-network budget per method (second seed, averaged predictions),
    # and try mixup on top of the relabel targets.
    "improve1": [
        ("t4_cvrelabel_s1", "--label noisy --seed 1 --targets_file {AUX}/targets_cvrelabel.npy"),
        ("t4_gce_s1", "--label noisy --seed 1 --loss gce"),
        ("t4_cvrelabel_mixup", "--label noisy --mixup 1.0 --targets_file {AUX}/targets_cvrelabel.npy"),
        ("t4_cvrelabel_mixup_s1", "--label noisy --seed 1 --mixup 1.0 --targets_file {AUX}/targets_cvrelabel.npy"),
    ],
    # Task 5 for the updated method set: best method (relabel + mixup, 2 nets) and the second GCE net,
    # on synthetic labels with targets rebuilt from the synthetic CV folds.
    "task5c": [
        ("t5_cvrelabel_mixup_synth", "--label synthetic --mixup 1.0 --targets_file {AUX}/targets_cvrelabel_synth.npy"),
        ("t5_cvrelabel_mixup_synth_s1", "--label synthetic --seed 1 --mixup 1.0 --targets_file {AUX}/targets_cvrelabel_synth.npy"),
        ("t5_gce_synth_s1", "--label synthetic --seed 1 --loss gce"),
    ],
    # Improvement round 2: mixup helped the relabel method, so try it with the robust loss too.
    "improve2": [
        ("t4_gce_mixup", "--label noisy --loss gce --mixup 1.0"),
        ("t4_gce_mixup_s1", "--label noisy --loss gce --mixup 1.0 --seed 1"),
    ],
    # Task 5 for the new best method (GCE + mixup, 2 nets).
    "task5d": [
        ("t5_gce_mixup_synth", "--label synthetic --loss gce --mixup 1.0"),
        ("t5_gce_mixup_synth_s1", "--label synthetic --loss gce --mixup 1.0 --seed 1"),
    ],
    # ---- Hyper-parameter sweep (noisy labels, seed 0, selection by NOISY validation accuracy only).
    # Reference single-net seed-0 runs: t4_gce_mixup (q .7, mixup 1) and t4_cvrelabel_mixup (beta .2, mixup 1).
    "sweepA": [  # robust loss + mixup
        ("s_gce_q05_mix1", "--label noisy --loss gce --gce_q 0.5 --mixup 1.0"),
        ("s_gce_q09_mix1", "--label noisy --loss gce --gce_q 0.9 --mixup 1.0"),
        ("s_gce_q07_mix05", "--label noisy --loss gce --mixup 0.5"),
        ("s_gce_q07_mix2", "--label noisy --loss gce --mixup 2.0"),
        ("s_sce_mix1", "--label noisy --loss sce --mixup 1.0"),
        ("s_ce_mix1", "--label noisy --mixup 1.0"),
    ],
    "sweepB": [  # relabel targets + mixup
        ("s_relabel_b0_mix1", "--label noisy --mixup 1.0 --targets_file {AUX}/targets_cvrelabel_b0.npy"),
        ("s_relabel_b05_mix1", "--label noisy --mixup 1.0 --targets_file {AUX}/targets_cvrelabel_b05.npy"),
        ("s_relabel_b02_mix2", "--label noisy --mixup 2.0 --targets_file {AUX}/targets_cvrelabel.npy"),
        ("s_relabel_b02_mix1_gce", "--label noisy --loss gce --mixup 1.0 --targets_file {AUX}/targets_cvrelabel.npy"),
        ("s_ls03_mix1", "--label noisy --label_smoothing 0.3 --mixup 1.0"),
    ],
    "sweepC": [  # two-network methods
        ("t4_dividemix", "--label noisy", "dividemix.py"),
        ("s_coteach_fr03", "--label noisy --forget_rate 0.3", "coteach.py"),
        ("s_coteach_fr04", "--label noisy --forget_rate 0.4", "coteach.py"),
    ],
    # second seeds for the two best relabel configs (2-net averages), and co-teaching fr 0.4 on synthetic labels
    "improve3": [
        ("t5_coteach_fr04_synth", "--label synthetic --forget_rate 0.4", "coteach.py"),
        ("s_relabel_b02_mix2_s1", "--label noisy --seed 1 --mixup 2.0 --targets_file {AUX}/targets_cvrelabel.npy"),
        ("s_relabel_b05_mix1_s1", "--label noisy --seed 1 --mixup 1.0 --targets_file {AUX}/targets_cvrelabel_b05.npy"),
    ],
    # Task 5 for the final best method: relabel (beta 0.2) + mixup alpha 2, 2 nets
    "task5e": [
        ("t5_cvrelabel_mix2_synth", "--label synthetic --mixup 2.0 --targets_file {AUX}/targets_cvrelabel_synth.npy"),
        ("t5_cvrelabel_mix2_synth_s1", "--label synthetic --seed 1 --mixup 2.0 --targets_file {AUX}/targets_cvrelabel_synth.npy"),
    ],
    # Optimizer comparison (not part of the submission): AdamW instead of SGD, same schedule shape.
    "adamw": [
        ("o_adamw_noisy_lr1e3", "--label noisy --optimizer adamw --lr 1e-3 --wd 0.05"),
        ("o_adamw_noisy_lr3e3", "--label noisy --optimizer adamw --lr 3e-3 --wd 0.05"),
        ("o_adamw_clean_lr2e3", "--label clean --optimizer adamw --lr 2e-3 --wd 0.05"),
        ("o_adamw_relabel_mix2", "--label noisy --optimizer adamw --lr 2e-3 --wd 0.05 --mixup 2.0 --targets_file {AUX}/targets_cvrelabel.npy"),
    ],
    # Forward loss correction with the transition matrix estimated WITHOUT clean labels (code/estimate_T.py).
    "forward": [
        ("s_fwd_ce", "--label noisy --forward_T {AUX}/T_est.npy"),
        ("s_fwd_mix1", "--label noisy --forward_T {AUX}/T_est.npy --mixup 1.0"),
        ("s_fwd_mix2", "--label noisy --forward_T {AUX}/T_est.npy --mixup 2.0"),
    ],
    # ELR and SOP (standard CIFAR-10 settings: ELR lambda 3, beta 0.7; SOP lr_u = lr_v = 10)
    "elrsop": [
        ("s_elr", "--label noisy --elr_lambda 3"),
        ("s_sop", "--label noisy --sop"),
        ("s_elr_mix1", "--label noisy --elr_lambda 3 --mixup 1.0"),
        ("s_elr_mix2", "--label noisy --elr_lambda 3 --mixup 2.0"),
    ],
    # second seeds -> 2-net averages
    "elrsop2": [
        ("s_elr_mix1_s1", "--label noisy --elr_lambda 3 --mixup 1.0 --seed 1"),
        ("s_elr_mix2_s1", "--label noisy --elr_lambda 3 --mixup 2.0 --seed 1"),
        ("s_sop_s1", "--label noisy --sop --seed 1"),
    ],
    # Task 5 for the new best method: ELR (lambda 3, beta .7) + mixup 1, 2 nets, synthetic labels
    "task5f": [
        ("t5_elr_mix1_synth", "--label synthetic --elr_lambda 3 --mixup 1.0"),
        ("t5_elr_mix1_synth_s1", "--label synthetic --elr_lambda 3 --mixup 1.0 --seed 1"),
    ],
    # two-network ELR (cross targets from weight-averaged nets), real and synthetic labels
    "elrplus": [
        ("t4_elrplus", "--label noisy", "elrplus.py"),
        ("t5_elrplus_synth", "--label synthetic", "elrplus.py"),
    ],
    # cat->dog bias fixes on top of ELR + mixup (single net screens, seed 0)
    "balA": [
        ("b_elr_da", "--label noisy --elr_lambda 3 --mixup 1.0 --dist_align"),
        ("b_elr_rs05", "--label noisy --elr_lambda 3 --mixup 1.0 --resample_power 0.5"),
        ("b_elr_rs10", "--label noisy --elr_lambda 3 --mixup 1.0 --resample_power 1.0"),
    ],
    "balB": [
        ("b_elr_cw10", "--label noisy --elr_lambda 3 --mixup 1.0 --class_weight_power 1.0"),
        ("b_elr_da_rs10", "--label noisy --elr_lambda 3 --mixup 1.0 --dist_align --resample_power 1.0"),
        ("b_elr_da_cw10", "--label noisy --elr_lambda 3 --mixup 1.0 --dist_align --class_weight_power 1.0"),
    ],
    # winner (ELR + mixup + distribution alignment): second net, and both nets on synthetic labels (Task 5)
    "balC": [
        ("b_elr_da_s1", "--label noisy --elr_lambda 3 --mixup 1.0 --dist_align --seed 1"),
        ("t5_elr_da_synth", "--label synthetic --elr_lambda 3 --mixup 1.0 --dist_align"),
    ],
    "balD": [
        ("t5_elr_da_synth_s1", "--label synthetic --elr_lambda 3 --mixup 1.0 --dist_align --seed 1"),
    ],
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


RUNNER = r'''
import glob, os, re, subprocess, time, json, torch

data_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/labels.csv", recursive=True)]
assert data_dirs, "labels.csv not found - did you add the data dataset to this notebook?"
DATA = data_dirs[0]
aux_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/AUX_MARKER.txt", recursive=True)]
AUX = aux_dirs[0] if aux_dirs else None  # optional second dataset with Task 4 inputs (targets, subsets)
print("aux:", AUX)
OUT = "/kaggle/working/runs"
n_gpu = torch.cuda.device_count()
print("data:", DATA, "| GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
assert n_gpu > 0, "no GPU - set Accelerator to 'GPU T4 x2' in the notebook settings"

pending = list(JOBS)
running = {}  # gpu -> (name, Popen, logfile)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
while pending or running:
    for g in range(n_gpu):  # start a job on every free GPU
        if g not in running and pending:
            name, extra, *rest = pending.pop(0)
            script = rest[0] if rest else "train.py"
            if "{AUX}" in extra or "{FILE:" in extra:
                assert AUX, "this job needs the aux dataset"
                extra = extra.replace("{AUX}", AUX)
            for fname in re.findall(r"\{FILE:([^}]+)\}", extra):  # inline the contents of an aux file
                extra = extra.replace("{FILE:" + fname + "}", open(f"{AUX}/{fname}").read().strip())
            log = f"{OUT}/{name}.log"
            cmd = f"python -u code/{script} --name {name} {extra} --data_dir {DATA} --out_dir {OUT}"
            print(f"[{(time.time()-t0)/60:5.1f} min] GPU{g} start: {cmd}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            running[g] = (name, subprocess.Popen(cmd, shell=True, env=env,
                                                 stdout=open(log, "w"), stderr=subprocess.STDOUT), log)
    time.sleep(60)
    for g, (name, p, log) in list(running.items()):
        lines = open(log).read().strip().splitlines()
        last = lines[-1][:160] if lines else ""
        print(f"[{(time.time()-t0)/60:5.1f} min] GPU{g} {name}: {last}", flush=True)
        if p.poll() is not None:
            print(f"    -> {name} finished with exit code {p.returncode}", flush=True)
            if p.returncode != 0:
                print("\n".join(lines[-30:]))
            del running[g]
print(f"all jobs done in {(time.time()-t0)/60:.1f} min")
'''

SUMMARY = r'''
import pandas as pd, json, glob, os
for s in sorted(glob.glob("/kaggle/working/runs/*/summary.json")):
    print(os.path.basename(os.path.dirname(s)), json.load(open(s)))
for name, *_ in JOBS:
    p = f"/kaggle/working/runs/{name}/log.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        cols = [c for c in ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"] if c in df]
        print(f"\n=== {name} (every 5th epoch) ===")
        print(df[cols].iloc[4::5].round(4).to_string(index=False))
'''

PLOT = r'''
import matplotlib.pyplot as plt, pandas as pd, os
p = "/kaggle/working/runs/t1_noisy/log.csv"
if os.path.exists(p):
    df = pd.read_csv(p)
    plt.figure(figsize=(8, 5))
    for c in ["train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]:
        plt.plot(df.epoch, df[c], label=c)
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.grid(alpha=.3); plt.legend(); plt.title("Noisy-label CE run")
    plt.savefig("/kaggle/working/runs/t1_noisy/dynamics.png", dpi=120); plt.show()
'''

def build(jobs_key, path):
    cells = [md(f"""
# CV bootcamp - {jobs_key}

Normally pushed and run with `python kaggle/kgl.py push {jobs_key}` (data input and GPU T4 x2 are set by
kernel-metadata.json). Outputs land in /kaggle/working/runs/<run name>/.

This notebook was generated by `kaggle/make_notebook.py`; the code cells are copies of `code/`.
""")]
    cells.append(code("!mkdir -p code\n!nvidia-smi --query-gpu=name,memory.total --format=csv"))
    for f in CODE_FILES:
        with open(os.path.join(CV, "code", f), encoding="utf-8") as fh:
            cells.append(code(f"%%writefile code/{f}\n" + fh.read()))
    cells.append(code("JOBS = " + json.dumps(JOBS[jobs_key], indent=1)))
    cells += [code(RUNNER), code(SUMMARY), code(PLOT)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                      "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", path)


def zip_data(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(CV, "data"))):
            z.write(os.path.join(CV, "data", f), f"data/{f}")
    print(f"wrote {path} ({os.path.getsize(path) / 2**20:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="task1", choices=list(JOBS))
    ap.add_argument("--zip_data", action="store_true")
    a = ap.parse_args()
    build(a.jobs, os.path.join(HERE, f"cv_{a.jobs}.ipynb"))
    if a.zip_data:
        zip_data(os.path.join(HERE, "cv-bootcamp-data.zip"))
