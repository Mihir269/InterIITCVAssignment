"""Task 3: score every training image (higher = noisy label more likely wrong).

Signals (all computed from models trained on NOISY labels only):
  cv_loss      5-fold cross-validation: each image is scored by the model that never trained on it,
               -log p(noisy label), at the epoch with the best NOISY validation accuracy for that fold.
  cv_loss_avg  the same, averaged over all epochs up to that best epoch (smoother).
  cv_margin    held-out margin: max other-class prob - p(noisy label), at the best epoch.
  insample_es  in-sample loss of the Task 1 noisy run at its early-stop epoch (no CV, 1 run).
  insample_aum mean in-sample margin over epochs up to the early-stop epoch ("area under the margin").
  rank_mix     average rank of cv_loss_avg and insample_aum.

Clean labels are used ONLY to report AUROC / precision@k of each signal (the allowed "own Task 3 check").

    python code/task3_scores.py --use cv_loss_avg      # writes predictions/t3_noise_scores.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")


def softmax(z):
    z = z.astype(np.float32)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def margin(p, y):
    """max other-class prob - prob of the given label (higher = label looks wrong)."""
    py = np.take_along_axis(p, y[..., None], -1)[..., 0]
    other = p.copy()
    np.put_along_axis(other, y[..., None], -1, -1)
    return other.max(-1) - py


def nll(p, y):
    return -np.log(np.clip(np.take_along_axis(p, y[..., None], -1)[..., 0], 1e-7, 1))


def best_epoch(run_dir):
    log = pd.read_csv(os.path.join(run_dir, "log.csv"))
    return int(log.loc[log.val_acc_given.idxmax(), "epoch"])  # noisy validation labels only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(CV, "outputs", "runs"))
    ap.add_argument("--data_dir", default=os.path.join(CV, "data"))
    ap.add_argument("--use", default="cv_loss_avg", help="which signal to write as the submission file")
    ap.add_argument("--out", default=os.path.join(CV, "predictions", "t3_noise_scores.csv"))
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(args.data_dir, "labels.csv")).set_index("index").sort_index()
    train_idx = np.loadtxt(os.path.join(args.data_dir, "train_idx.txt"), dtype=np.int64)
    noisy = labels.noisy_label.values
    signals = {}
    cv_probs = np.zeros((len(labels), 10), np.float32)  # held-out probs per image, for Task 4 relabeling

    # ---- cross-validated signals
    fold_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "t3_cv*")) if os.path.isdir(d))
    if fold_dirs:
        s_loss, s_avg, s_margin = (np.full(len(labels), np.nan) for _ in range(3))
        for d in fold_dirs:
            e = best_epoch(d)
            idx = np.load(os.path.join(d, "holdout_index.npy"))
            P = softmax(np.load(os.path.join(d, "holdout_logits.npy"))[:e])  # epochs 1..e
            y = noisy[idx]
            s_loss[idx] = nll(P[-1], y)
            s_avg[idx] = nll(P, np.broadcast_to(y, P.shape[:2])).mean(0)
            s_margin[idx] = margin(P[-1], y)
            cv_probs[idx] = P[-1]
            print(f"{os.path.basename(d)}: best noisy-val epoch {e}, {len(idx)} held-out images")
        assert not np.isnan(s_loss[train_idx]).any(), "some training images have no held-out score"
        signals |= dict(cv_loss=s_loss, cv_loss_avg=s_avg, cv_margin=s_margin)
        os.makedirs(os.path.join(CV, "outputs", "task3"), exist_ok=True)
        np.save(os.path.join(CV, "outputs", "task3", "cv_probs.npy"), cv_probs)

    # ---- in-sample signals from the Task 1 noisy run
    t1 = os.path.join(args.runs, "t1_noisy")
    if os.path.exists(os.path.join(t1, "train_logits.npy")):
        e = best_epoch(t1)
        idx = np.load(os.path.join(t1, "train_index.npy"))
        L = np.load(os.path.join(t1, "train_logits.npy"), mmap_mode="r")
        y = noisy[idx]
        s_es, s_aum = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
        s_es[idx] = nll(softmax(L[e - 1]), y)
        s_aum[idx] = np.mean([margin(softmax(L[k]), y) for k in range(e)], 0)
        signals |= dict(insample_es=s_es, insample_aum=s_aum)
        print(f"t1_noisy: early-stop epoch {e}")

    if "cv_loss_avg" in signals and "insample_aum" in signals:
        r = lambda s: pd.Series(s[train_idx]).rank(pct=True).values  # noqa: E731
        mix = np.full(len(labels), np.nan)
        mix[train_idx] = (r(signals["cv_loss_avg"]) + r(signals["insample_aum"])) / 2
        signals["rank_mix"] = mix

    # ---- the allowed check against clean labels (measurement only)
    wrong = (labels.noisy_label.values != labels.clean_label.values)[train_idx]
    k = int(wrong.sum())
    rows = []
    for name, s in signals.items():
        st = s[train_idx]
        top = np.argsort(-st, kind="stable")[:k]
        rows.append(dict(signal=name, auroc=roc_auc_score(wrong, st), precision_at_k=wrong[top].mean()))
    res = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print(f"\nk = true number of mislabeled training images = {k}")
    print(res.round(4).to_string(index=False))
    os.makedirs(os.path.join(CV, "outputs", "task3"), exist_ok=True)
    res.to_csv(os.path.join(CV, "outputs", "task3", "signal_comparison.csv"), index=False)

    if args.use not in signals:
        raise SystemExit(f"signal '{args.use}' not available; have {list(signals)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame({"index": train_idx, "score": signals[args.use][train_idx]}).to_csv(args.out, index=False)
    np.save(os.path.join(CV, "outputs", "task3", "scores.npy"), signals[args.use])
    print(f"\nwrote {args.out} using '{args.use}'")
    with open(os.path.join(CV, "outputs", "task3", "chosen.json"), "w") as f:
        json.dump(dict(signal=args.use), f)


if __name__ == "__main__":
    main()
