# Experiment log

Everything that was tried, in order, with results and the reason for each decision. Per-epoch logs of every
run are in `results/runs/<run>/log.csv`; ranked tables in `results/tables/`.

**Reading the numbers.** "Noisy val" = accuracy on the 5,000 validation images against their *noisy* labels –
the only signal used to select models (synthetic labels in Task 5). "Clean val" = accuracy against the true
labels – a **measurement only**, never used to train, filter, relabel or select. Because ~40% of the noisy
validation labels are wrong, even a perfect model scores only ~59–60% noisy val (the clean-label reference
model scores 59.1%), and one noisy-val number is uncertain by about ±0.7 points (n = 5,000).

---

## 0. Setup decisions

| decision | reason |
|---|---|
| ResNet-18 from `model.py`, from scratch, 60 epochs, seed 0 | problem statement |
| SGD + Nesterov 0.9, wd 5e-4, batch 128, OneCycle cosine LR (peak 0.1), crop + flip | standard CIFAR recipe, fixed **before** any experiment and never tuned (tuning it on noisy labels would make the baseline implicitly noise-robust) |
| Mixed precision, data and augmentation on the GPU | speed: ~17 s/epoch on a Tesla T4 |
| Save val/test/train logits every epoch | any epoch can be selected afterwards without retraining |
| Kaggle (2× T4) driven from the local machine by `code/kaggle/` | the local RTX 2050 needed ~4 min/epoch |

## 1. The noise (Task 1)

* 40.24% of training labels are wrong (18,109 / 45,000); validation 39.9%.
* Human errors concentrate on look-alike pairs (both directions): automobile↔truck 13.1% of errors,
  cat↔dog 9.7%, deer↔horse 5.1%, airplane↔ship 4.1% (32% together). Cat and deer are 51% wrong, horse 28%.
* Synthetic labels: same rate, spread evenly (~4–5% into every other class).
  Figure: `results/figures/confusion_human_vs_synthetic.png`.

## 2. Baselines and memorization (Tasks 1–2)

| run | clean val | noisy val |
|---|---|---|
| clean-label reference, final epoch | **94.5%** | 59.1% |
| noisy CE, final epoch (60) | 70.5% | 50.4% |
| noisy CE, early stop (epoch 43, max noisy val) | **77.1%** | 55.9% |

Memorization starts at **epoch 38** (first epoch after which accuracy on the mislabeled subset stays above
mean + 3σ of its first-half level): agreement with wrong labels is flat at 12–15% until ~37, then climbs to 64%
at epoch 60, while the share of mislabeled images predicted as their true class falls from ~65% to 25%.
Figure: `results/figures/memorization_dynamics.png`.

## 3. Finding mislabeled images (Task 3)

Five-fold cross-validation on noisy labels (the 5-run budget): each image scored by the fold model that never
trained on it. AUROC / precision@k (k = 18,109) are the allowed clean-label check.

| signal | AUROC | P@k |
|---|---|---|
| **AUM** – margin (max other prob − prob of noisy label), Task 1 run, epochs 1–43 (**submitted**) | **0.966** | **0.882** |
| AUM + CV loss, rank average | 0.959 | 0.862 |
| CV held-out loss, averaged over epochs ≤ best | 0.938 | 0.815 |
| in-sample loss at the early-stop epoch | 0.930 | 0.825 |
| CV held-out loss / margin at the best epoch | 0.925 / 0.924 | 0.811 / 0.819 |

Failure mode of AUM: recall 79% on automobile↔truck and cat↔dog errors vs 91% on all other errors.

Other ideas (not submitted):

| idea | AUROC | why not submitted |
|---|---|---|
| kNN label disagreement (50 neighbours) on features of a CE+mixup model | 0.929 | no gain over AUM (0.964 combined) |
| same on features of the best Task 4 model (relabel+mixup) | 0.967 (0.973 with AUM) | needs a Task 4 model → a 6th model feeding Task 3 (budget is 5 runs) and circular; gain tiny |
| same on features of the memorized CE model | 0.750 | memorization pulls wrong images to their wrong class |
| per-class centroid / k-means distance (best features) | 0.926 / 0.903 | below kNN |
| hand-crafted features (LBP, GLCM, HOG, colour), kNN | 0.825 (HOG alone 0.804) | texture cannot separate cat/dog at 32×32; any weight in a mix with AUM lowered AUROC |
| look-alike pairs in hand-crafted space, higher-loss member flagged | 0.913 | only 49% of hand-crafted neighbours share the true class; plain loss alone scores 0.949 |
| late "memorization jump" in loss | 0.765 | correct hard images also drop late |

## 4. Robust training (Task 4)

Budget per method: ≤ 2 networks × 60 epochs, same recipe. Prediction file = epoch with the best noisy val.

### Round 1 – one network each
| method | noisy val | clean val |
|---|---|---|
| GCE (q 0.7) | 57.14 | 81.0 |
| CV relabel (targets 0.2·noisy one-hot + 0.8·held-out CV prediction; 78.5% of targets point to the true class vs 59.8% of noisy labels) | 57.88 | 82.7 |
| co-teaching (forget rate 0.466 = CV disagreement estimate) | 57.78 | 85.4 |

### Round 2 – full 2-network budget and mixup
| method | noisy val | clean val |
|---|---|---|
| GCE, 2 nets averaged | 58.50 | 84.3 |
| relabel, 2 nets | 58.04 | 82.9 |
| relabel + mixup 1, 2 nets | 58.72 | 85.8 |
| GCE + mixup 1, 2 nets | 58.96 | 85.1 |

### Round 3 – sweep (single net, seed 0 unless noted)
| config | noisy val |
|---|---|
| relabel β 0.2 + mixup 2 → 2 nets | 58.94 → **59.14** |
| relabel β 0.5 + mixup 1 → 2 nets | 58.88 → 59.12 |
| co-teaching forget 0.4 / 0.3 (2 nets) | 58.52 / 58.08 |
| relabel targets + GCE + mixup 1 | 58.32 |
| relabel β 0 + mixup 1 | 58.18 |
| GCE + mixup 2 / 0.5 | 57.98 / 57.98 |
| GCE q 0.5 / 0.9 + mixup 1 | 57.80 / 56.82 |
| label smoothing 0.3 + mixup 1 | 57.78 |
| CE + mixup 1 | 57.44 |
| SCE + mixup 1 | 55.72 |
| **DivideMix-style** (2 nets, 10 warm-up epochs) | 58.26 (clean val **89.3**) |

Finding: the top configurations differ by less than the ±0.7-point noise of noisy val, and the method with
the best clean accuracy (DivideMix) ranks lower on noisy val, because it predicts the true class where the
noisy validation label is wrong. It was **not** submitted: choosing it would mean selecting with clean labels.

### Side experiments
| experiment | result | decision |
|---|---|---|
| AdamW (lr 1e-3 / 3e-3, wd 0.05) instead of SGD | clean reference 93.8% (SGD 94.5%); noisy CE memorizes from ~epoch 20 (87% vs 64% fit to wrong labels at epoch 60, final clean 63.9% vs 70.5%); relabel+mixup ties (59.00) | keep SGD |
| transition matrix T estimated **without clean labels** (anchor points from held-out CV predictions; mean abs. error 0.024) + forward loss correction | CE 56.10 (final clean 76.1% vs 70.5%), + mixup 1 / 2: 57.54 / 57.84 | below mixup methods |
| noise-corrected validation (invert T on the noisy validation confusion) | ranks runs no better than plain noisy val (Spearman 0.835 vs 0.846), even with the true T it overestimates accuracy (~98% for ~85% models) | unusable: human errors are instance-dependent |

### Round 4 – ELR and SOP
| method | noisy val | clean val |
|---|---|---|
| ELR (λ 3, β 0.7), 1 net | 58.50 | 87.2 |
| ELR + mixup 1, 1 net → **2 nets** | 58.86 → **59.22** | 88.7 → **89.5** |
| ELR + mixup 2, 1 net → 2 nets | 58.82 → 58.84 | 88.5 → 89.2 |
| SOP (per-sample noise terms), 1 net → 2 nets | 58.66 → 58.92 | 83.7 → 85.7 |
| two-network ELR (cross targets from weight-averaged nets, mixup 1) | 58.84 | 88.9 |

### Round 5 – the cat→dog bias
Annotators over-use "dog" and under-use "cat"; ELR + mixup inherits it (test predictions: 1,359 dogs, 668 cats;
clean val cat 64% vs dog 94%). Fixes that use no labels:

| ELR + mixup + … (single net, seed 0) | noisy val | clean val | cat / dog acc (measured) |
|---|---|---|---|
| nothing (reference) | 58.86 | 88.7 | 62.7 / 93.4 |
| **distribution alignment** (ELR targets rescaled by uniform / running-mean prediction) | **59.32** | **90.3** | **78.7** / 86.1 |
| rebalanced sampling, power 0.5 / 1.0 (rare labels drawn more often, fresh crop/flip each time) | 59.16 / 59.14 | 89.8 / 89.7 | 68.9 / 76.8 |
| class-weighted loss (inverse label frequency) | 58.92 | 89.3 | 70.8 / 89.1 |
| distribution alignment + sampling 1.0 / + class weights | 58.96 / 58.84 | 90.1 / 90.4 | 77.0 / 81.0 |

Distribution alignment, 2 nets: 59.24 noisy val, 90.56 clean (seed 1 alone: 58.84 / 90.20).

**Post-hoc class rebalancing** (`code/rebalance.py`, no retraining): class shares of the networks' own predictions
on the training images (ELR + mixup: cat 6.9%, dog 13.4%), outputs shifted by −τ·log(share / 0.1); τ chosen from
{0, 0.25, …, 1.5} by noisy validation:

| method (2 nets) | τ chosen | noisy val | clean val |
|---|---|---|---|
| ELR + mixup | 0.75 | 59.22 → **59.30** | 89.46 → 89.90 (cat 64% → 68%) |
| CV relabel + mixup | 0.5 | 59.14 → 59.26 | 86.26 → 87.74 |
| GCE + mixup | 0 | 58.96 (unchanged) | 85.06 |
| ELR + mixup + distribution alignment | 0 | 59.24 (unchanged) | 90.56 |

### Final selection
The three methods with the highest noisy validation accuracy (2 networks each; post-hoc rebalancing is part of
each method, with τ chosen on noisy val):

| method | noisy val | clean val (measured) |
|---|---|---|
| **ELR + mixup + class rebalancing (`best_method`)** | **59.30** | 89.9 |
| CV relabel + mixup + class rebalancing | 59.26 | 87.7 |
| ELR + mixup + distribution alignment | 59.24 | **90.6** |
| *(not submitted)* ELR + mixup | 59.22 | 89.5 |
| *(not submitted)* GCE + mixup | 58.96 | 85.1 |
| *(not submitted)* SOP | 58.92 | 85.7 |
| *(not submitted)* co-teaching forget 0.4 | 58.52 | 85.6 |
| *(not submitted)* DivideMix | 58.26 | 89.3 |

The top four are within 4 images of each other on noisy validation (±0.7-point noise), so the ranking among them
is essentially a tie; the rule-based choice still submits the most accurate model (distribution alignment) as the
third method.

## 5. Real vs synthetic noise (Task 5)

| method | real: noisy val | real: clean | synthetic: synth val | synthetic: clean |
|---|---|---|---|---|
| CE (early stop; Task 5 file is the final epoch: 45.4 / 74.0) | 55.92 | 77.1 | 50.22 | 83.3 |
| ELR + mixup + rebalancing (τ 0.5 on synthetic) | 59.30 | 89.9 | 54.66 | 92.5 |
| CV relabel + mixup + rebalancing (τ 1.25 on synthetic) | 59.26 | 87.7 | 54.28 | 91.5 |
| ELR + mixup + distribution alignment | 59.24 | 90.6 | 54.56 | 92.6 |

ELR + rebalancing leads on both; the other two swap places within noise. Synthetic noise is easier for every
method; the methods gain more on human noise (+12.8 vs +9.2 clean points over CE). On synthetic labels the
networks' predictions are already balanced (cat 9.3%, dog 9.9% of training images), so rebalancing changes almost
nothing. Earlier ablation: mixup helped GCE on human noise (+0.46) but not on synthetic noise (−0.36).

**Why human noise is harder** (`code/noise_difficulty.py`; difficulty = confidence of the clean-label reference
model in the true class on validation images it never trained on):

| hardest q% | human: mislabeled among hardest / rest | random: hardest / rest |
|---|---|---|
| 10% | 59% / 38% | 43% / 41% |
| 20% | 52% / 37% | 44% / 40% |
| 30% | 51% / 35% | 42% / 41% |
| 50% | 46% / 34% | 42% / 40% |

Humans mislabel the ambiguous images (and only look-alike classes); random noise ignores both. Human wrong
labels therefore look plausible to the network, are memorized more easily, are harder to detect and cannot be
undone with a per-class transition matrix.

## 6. Infrastructure notes
* First Kaggle dataset upload (loose files) failed to mount on Kaggle (`ERRORED_MOUNTING_DATASET`); re-uploading
  the data as a single zip fixed it.
* Kaggle allows 2 concurrent GPU sessions; `code/kaggle/` queues jobs, one per GPU.
* About 30 GPU-hours on Kaggle in total.
