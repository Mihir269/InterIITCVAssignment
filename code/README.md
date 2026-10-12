# `code/` – scripts and exact commands

All commands are run from the repository root, with the provided bootcamp files in `data/`
(see `data/README.md`). Every network is the provided ResNet-18 (`code/model.py`, unchanged), trained from
scratch for 60 epochs, seed 0, with one fixed recipe:

* SGD, momentum 0.9 (Nesterov), weight decay 5e-4, batch 128, mixed precision (fp16)
* OneCycle learning rate per batch: cosine warm-up 0.01 → 0.1 over the first 15% of steps, cosine decay to 1e-5
* augmentation: random crop (4-px zero padding) + horizontal flip, on the GPU

Model / epoch selection always uses `val_acc_given` = validation accuracy against the label column the model
was trained on (noisy for Tasks 1–4, synthetic for Task 5). Clean labels are used only to train the Task 1
reference model and for logged measurements (`val_acc_clean`, `train_acc_mislabeled`, AUROC checks).

Each run writes `outputs/runs/<name>/`: `config.json`, `log.csv` (one row per epoch), `pred_final.csv`,
`pred_best.csv`, per-epoch logits and final weights. `outputs/` is not committed; `results/` holds a small
snapshot (`python code/collect_results.py --dest results`).

## Scripts

| script | purpose |
|---|---|
| `model.py` | the provided ResNet-18 (unchanged copy) |
| `train.py` | single-network trainer: `--label clean/noisy/synthetic`; `--loss ce/gce/sce`; `--mixup`; `--label_smoothing`; `--elr_lambda` (ELR); `--sop` (SOP); `--forward_T` (forward correction); `--targets_file` (relabel targets); `--cv_folds/--cv_fold` (Task 3 folds); `--optimizer adamw` (comparison only) |
| `ensemble.py` | average two runs of a method (the 2-network budget) epoch by epoch; best epoch by `val_acc_given` |
| `coteach.py`, `dividemix.py`, `elrplus.py` | two-network methods (co-teaching, DivideMix-style, two-network ELR) |
| `analyze_noise.py` | Task 1: noise rate, 10x10 noisy-vs-clean matrices, figure |
| `task2_analyze.py` | Task 2: memorization start, early-stop epoch, figure, t1/t2 prediction files |
| `task3_scores.py` | Task 3: six noise signals compared, writes `predictions/t3_noise_scores.csv` |
| `task4_targets.py` | relabel targets + noise-rate estimate from the 5 Task 3 fold models |
| `feature_scores.py`, `handcrafted_scores.py`, `pair_scores.py` | Task 3 experiments: kNN / centroid scores on network features, hand-crafted (LBP, GLCM, HOG, colour) features, look-alike pairs |
| `estimate_T.py`, `noise_corrected_val.py` | transition matrix estimated without clean labels; noise-corrected validation |
| `noise_difficulty.py` | are mislabeled images harder? (human vs synthetic, Q5) |
| `data_analysis.py` | final data analysis: figures + numbers for docs/ANALYSIS.md |
| `compare_runs.py`, `collect_results.py` | rank runs; build the `results/` snapshot |
| `predict.py` | test predictions from the released weights in `models/` (1 or 2 networks) |
| `make_submission.py` | fills `results.json`, copies prediction files, runs `check_format.py`, builds the zip |
| `kaggle/kgl.py`, `kaggle/make_notebook.py` | run the same commands on Kaggle GPUs (2× T4) via the Kaggle CLI |

## Task 1 – data and baselines
```bash
python code/analyze_noise.py
python code/train.py --label clean --name t1_clean      # -> predictions/t1_clean_ref.csv (final epoch)
python code/train.py --label noisy --name t1_noisy      # -> predictions/t1_noisy_ce.csv  (final epoch)
```

## Task 2 – training dynamics
```bash
python code/task2_analyze.py      # t2_dynamics.csv, t2_early_stop.csv (epoch 43), memorization start (38)
```

## Task 3 – noise scores (5 training runs)
```bash
for k in 0 1 2 3 4; do python code/train.py --label noisy --name t3_cv$k --cv_folds 5 --cv_fold $k --no_train_logits; done
python code/task3_scores.py --use insample_aum          # submitted: AUM from the Task 1 noisy run
# experiments
python code/feature_scores.py --source s_ce_mix1        # also: --source cv | t1_noisy | s_relabel_b02_mix2
python code/handcrafted_scores.py
python code/pair_scores.py
```

## Task 4 – robust training (≤ 2 networks × 60 epochs per method)
```bash
python code/task4_targets.py     # relabel targets (beta 0.2) and estimated noise rate from the Task 3 folds

# submitted 1/3 – elr_mixup (best): ELR lambda 3, target momentum 0.7, mixup alpha 1, 2 nets averaged
python code/train.py --label noisy --name s_elr_mix1    --elr_lambda 3 --mixup 1.0
python code/train.py --label noisy --name s_elr_mix1_s1 --elr_lambda 3 --mixup 1.0 --seed 1
python code/ensemble.py --runs s_elr_mix1 s_elr_mix1_s1 --name t4_elr_mix1_ens

# submitted 2/3 – cv_relabel_mixup: targets 0.2*noisy one-hot + 0.8*held-out CV prediction, mixup alpha 2
python code/train.py --label noisy --name s_relabel_b02_mix2    --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel.npy
python code/train.py --label noisy --name s_relabel_b02_mix2_s1 --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel.npy --seed 1
python code/ensemble.py --runs s_relabel_b02_mix2 s_relabel_b02_mix2_s1 --name t4_cvrelabel_mix2_ens

# submitted 3/3 – gce_mixup: GCE q 0.7, mixup alpha 1
python code/train.py --label noisy --name t4_gce_mixup    --loss gce --mixup 1.0
python code/train.py --label noisy --name t4_gce_mixup_s1 --loss gce --mixup 1.0 --seed 1
python code/ensemble.py --runs t4_gce_mixup t4_gce_mixup_s1 --name t4_gce_mixup_ens

# examples of the other experiments (full list with results in docs/EXPERIMENTS.md)
python code/train.py --label noisy --name t4_gce --loss gce
python code/train.py --label noisy --name s_sop --sop
python code/coteach.py --label noisy --name s_coteach_fr04 --forget_rate 0.4
python code/dividemix.py --label noisy --name t4_dividemix
python code/elrplus.py --label noisy --name t4_elrplus
python code/estimate_T.py && python code/train.py --label noisy --name s_fwd_mix2 --forward_T outputs/aux/T_est.npy --mixup 2.0
python code/noise_corrected_val.py
python code/compare_runs.py                 # rank every run by noisy validation accuracy
```

## Task 5 – synthetic noise
```bash
python code/train.py --label synthetic --name t5_baseline_synth                       # -> t5_baseline_synth.csv (final epoch)
python code/train.py --label synthetic --name t5_elr_mix1_synth    --elr_lambda 3 --mixup 1.0
python code/train.py --label synthetic --name t5_elr_mix1_synth_s1 --elr_lambda 3 --mixup 1.0 --seed 1
python code/ensemble.py --runs t5_elr_mix1_synth t5_elr_mix1_synth_s1 --name t5_elr_mix1_synth_ens   # -> t5_best_synth.csv
# the other submitted methods on synthetic labels (Q5 ranking)
for k in 0 1 2 3 4; do python code/train.py --label synthetic --name t5_cv$k --cv_folds 5 --cv_fold $k --no_train_logits; done
python code/task4_targets.py --label synthetic --prefix t5_cv --suffix _synth
python code/train.py --label synthetic --name t5_cvrelabel_mix2_synth    --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel_synth.npy
python code/train.py --label synthetic --name t5_cvrelabel_mix2_synth_s1 --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel_synth.npy --seed 1
python code/ensemble.py --runs t5_cvrelabel_mix2_synth t5_cvrelabel_mix2_synth_s1 --name t5_cvrelabel_mix2_synth_ens
python code/train.py --label synthetic --name t5_gce_mixup_synth    --loss gce --mixup 1.0
python code/train.py --label synthetic --name t5_gce_mixup_synth_s1 --loss gce --mixup 1.0 --seed 1
python code/ensemble.py --runs t5_gce_mixup_synth t5_gce_mixup_synth_s1 --name t5_gce_mixup_synth_ens
python code/noise_difficulty.py
```

## Assemble the submission
```bash
python code/make_submission.py --roll_no 24b2443 --name "Mihir Gune" --t5_best_run t5_elr_mix1_synth_ens
```

## Running on Kaggle GPUs from a local machine
```bash
python code/kaggle/kgl.py data          # once: upload data/ as a private Kaggle dataset
python code/kaggle/kgl.py push task1    # run a job list from code/kaggle/make_notebook.py on 2x T4
python code/kaggle/kgl.py status task1
python code/kaggle/kgl.py fetch task1   # download into outputs/runs/
```
