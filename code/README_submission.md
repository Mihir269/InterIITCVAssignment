# CIFAR-10N (worst labels) – reproduction guide

Full repository (all experiments, per-run logs, figures, detailed experiment log):
https://github.com/Mihir269/InterIITCVAssignment

All commands are run from the folder that contains `data/` (the provided bootcamp folder), with this
`code/` folder inside it. Every network is the provided ResNet-18 (`code/model.py`, an unchanged copy),
trained from scratch, 60 epochs, seed 0, same recipe everywhere:

* SGD, momentum 0.9 (Nesterov), weight decay 5e-4, batch 128, mixed precision (fp16)
* OneCycle LR per batch: cosine warm-up 0.01 -> 0.1 over the first 15% of steps, cosine decay to 1e-5
* augmentation: random crop (4-px zero padding) + horizontal flip, done on the GPU

Clean labels are used only (a) as training labels for the Task 1 reference model and (b) for logged
measurements (Task 1 confusion matrix, Task 2 curves, the Task 3 AUROC check). Model/epoch selection uses
`val_acc_given` = validation accuracy against the training label column (noisy for Tasks 1-4, synthetic
for Task 5).

Each `train.py` run writes `outputs/runs/<name>/` with `log.csv`, `pred_final.csv` (last epoch),
`pred_best.csv` (best validation epoch on the training labels), and per-epoch logits.

Requirements: Python 3.10+, PyTorch 2.x with CUDA, numpy, pandas, scikit-learn, matplotlib.
A Tesla T4 takes ~17 s per epoch (~18 min per 60-epoch run).

## Task 1 – data and baselines
```bash
python code/analyze_noise.py                       # noise rate + 10x10 confusion matrices (outputs/analysis/)
python code/train.py --label clean --name t1_clean # clean reference   -> t1_clean_ref.csv = pred_final.csv
python code/train.py --label noisy --name t1_noisy # noisy CE baseline -> t1_noisy_ce.csv  = pred_final.csv
```

## Task 2 – training dynamics
The `t1_noisy` run logs the four curves every epoch (`dynamics.csv`).
```bash
python code/task2_analyze.py    # memorization start, early-stop epoch (max NOISY val acc), plot, t1_*/t2_* files
```

## Task 3 – noise scores (5 training runs)
```bash
for k in 0 1 2 3 4; do python code/train.py --label noisy --name t3_cv$k --cv_folds 5 --cv_fold $k --no_train_logits; done
python code/task3_scores.py --use insample_aum   # compares 6 signals (AUROC check), writes predictions/t3_noise_scores.csv
# insample_aum uses the Task 1 noisy run (train_logits.npy); the 5 CV runs give the held-out signals and the Task 4 targets
```

Also measured (not submitted): feature-space outlier scores, `python code/feature_scores.py --source <run|cv>`
(kNN label disagreement among 50 nearest neighbours, per-class centroid / k-means distance on 512-d penultimate
features) and `python code/handcrafted_scores.py` (LBP, GLCM, HOG, colour histograms). kNN reaches AUROC 0.929
on CE+mixup features and 0.967 on relabel+mixup features, but only 0.750 on the memorized CE model and 0.825 on
hand-crafted features; adding hand-crafted scores to AUM never helped. The submitted file stays AUM, which
needs no extra trained model (Task 3 is limited to 5 training runs; the best kNN features come from a Task 4 model).

## Task 4 – robust training (same recipe and budget as the noisy CE baseline)
Every method uses at most 2 networks x 60 epochs; the test file is the epoch with the best NOISY validation
accuracy (for 2-net methods: of the averaged softmax). The post-hoc class rebalancing strength is chosen the same
way. Submitted = the three methods with the highest noisy validation accuracy (no clean labels in any choice).
```bash
python code/task4_targets.py     # relabel targets (beta 0.2) and estimated noise rate from the Task 3 folds

# submitted 1/3 – elr_mixup_rebalanced (best): ELR lambda 3, target momentum 0.7, mixup alpha 1, 2 nets, + rebalancing
python code/train.py --label noisy --name s_elr_mix1    --elr_lambda 3 --mixup 1.0
python code/train.py --label noisy --name s_elr_mix1_s1 --elr_lambda 3 --mixup 1.0 --seed 1
python code/ensemble.py --runs s_elr_mix1 s_elr_mix1_s1 --name t4_elr_mix1_ens
python code/rebalance.py --ens t4_elr_mix1_ens --members s_elr_mix1 s_elr_mix1_s1 --name t4_elr_mix1_bal

# submitted 2/3 – cv_relabel_mixup_rebalanced: targets 0.2*noisy one-hot + 0.8*held-out CV prediction, mixup alpha 2
python code/train.py --label noisy --name s_relabel_b02_mix2    --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel.npy
python code/train.py --label noisy --name s_relabel_b02_mix2_s1 --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel.npy --seed 1
python code/ensemble.py --runs s_relabel_b02_mix2 s_relabel_b02_mix2_s1 --name t4_cvrelabel_mix2_ens
python code/rebalance.py --ens t4_cvrelabel_mix2_ens --members s_relabel_b02_mix2 s_relabel_b02_mix2_s1 --name t4_cvrelabel_mix2_bal

# submitted 3/3 – elr_mixup_distalign: ELR + mixup with distribution-aligned ELR targets
python code/train.py --label noisy --name b_elr_da    --elr_lambda 3 --mixup 1.0 --dist_align
python code/train.py --label noisy --name b_elr_da_s1 --elr_lambda 3 --mixup 1.0 --dist_align --seed 1
python code/ensemble.py --runs b_elr_da b_elr_da_s1 --name t4_elr_da_ens
```

Noisy val of the submitted methods: elr_mixup_rebalanced 59.30 (best), cv_relabel_mixup_rebalanced 59.26,
elr_mixup_distalign 59.24. Other configurations tried (noisy val, 2 nets unless noted): ELR + mixup 59.22,
relabel + mixup 59.14, GCE + mixup 58.96, SOP 58.92, two-network ELR 58.84, co-teaching 58.52, DivideMix 58.26,
forward correction + mixup 57.84, ELR + rebalanced sampling 59.16 (1 net), ELR + class weights 58.92 (1 net),
plus ~30 loss / mixup / relabelling variants. Full table, per-epoch logs and scripts for every experiment:
https://github.com/Mihir269/InterIITCVAssignment (docs/EXPERIMENTS.md).

Note: the clean-label validation accuracy logged for these runs is a measurement only. DivideMix, for example,
measured 89.3% clean val but was not submitted, because its noisy validation accuracy was lower and clean labels
may not be used to select models.

## Task 5 – synthetic noise (same commands with --label synthetic)
```bash
python code/train.py --label synthetic --name t5_baseline_synth                       # -> t5_baseline_synth.csv (final epoch)
# best Task 4 method on synthetic labels -> t5_best_synth.csv
python code/train.py --label synthetic --name t5_elr_mix1_synth    --elr_lambda 3 --mixup 1.0
python code/train.py --label synthetic --name t5_elr_mix1_synth_s1 --elr_lambda 3 --mixup 1.0 --seed 1
python code/ensemble.py --runs t5_elr_mix1_synth t5_elr_mix1_synth_s1 --name t5_elr_mix1_synth_ens
python code/rebalance.py --ens t5_elr_mix1_synth_ens --members t5_elr_mix1_synth t5_elr_mix1_synth_s1 --name t5_elr_mix1_synth_bal
# the other submitted methods on synthetic labels (Q5 ranking)
for k in 0 1 2 3 4; do python code/train.py --label synthetic --name t5_cv$k --cv_folds 5 --cv_fold $k --no_train_logits; done
python code/task4_targets.py --label synthetic --prefix t5_cv --suffix _synth
python code/train.py --label synthetic --name t5_cvrelabel_mix2_synth    --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel_synth.npy
python code/train.py --label synthetic --name t5_cvrelabel_mix2_synth_s1 --mixup 2.0 --targets_file outputs/aux/targets_cvrelabel_synth.npy --seed 1
python code/ensemble.py --runs t5_cvrelabel_mix2_synth t5_cvrelabel_mix2_synth_s1 --name t5_cvrelabel_mix2_synth_ens
python code/rebalance.py --ens t5_cvrelabel_mix2_synth_ens --members t5_cvrelabel_mix2_synth t5_cvrelabel_mix2_synth_s1 --name t5_cvrelabel_mix2_synth_bal
python code/train.py --label synthetic --name t5_elr_da_synth    --elr_lambda 3 --mixup 1.0 --dist_align
python code/train.py --label synthetic --name t5_elr_da_synth_s1 --elr_lambda 3 --mixup 1.0 --dist_align --seed 1
python code/ensemble.py --runs t5_elr_da_synth t5_elr_da_synth_s1 --name t5_elr_da_synth_ens
python code/noise_difficulty.py
```

## Human vs random (synthetic) noise – analysis behind Q1 and Q5
```bash
python code/analyze_noise.py      # 10x10 matrices (true class -> given label) for human and synthetic labels
python code/noise_difficulty.py   # are mislabeled images harder? (measurement only)
```
1. **Which classes** (the 10x10 matrices written by `analyze_noise.py`): human errors pile onto look-alike pairs (automobile↔truck 13.1% of
   all errors, cat↔dog 9.7%, deer↔horse 5.1%, airplane↔ship 4.1%); synthetic errors are spread evenly
   (~4–5% into every other class).
2. **Which images**: difficulty of a validation image = confidence of the clean-label reference model
   (never trained on validation images) in the true class; "hard" = lowest q% of that confidence.

   | hardest q% | human: mislabeled among hardest / rest | random: hardest / rest |
   |---|---|---|
   | 10% | 59% / 38% | 43% / 41% |
   | 20% | 52% / 37% | 44% / 40% |
   | 30% | 51% / 35% | 42% / 41% |
   | 50% | 46% / 34% | 42% / 40% |

   Humans mislabel the ambiguous images; random noise ignores difficulty. This is why human wrong labels
   look plausible to the network, are memorized more easily, are harder to detect (Task 3 recall 79% on
   look-alike pairs vs 91% elsewhere) and cannot be undone with a per-class transition matrix.

## Assemble and check
```bash
python code/make_submission.py --roll_no <rollno> --name "<name>" --t5_best_run t5_elr_mix1_synth_bal
```
