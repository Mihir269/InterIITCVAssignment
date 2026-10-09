"""Task 3 (extra): pairwise look-alike check with hand-crafted features + the Task 1 model's losses.

  1. For every training image, find its nearest neighbours in hand-crafted feature space
     (LBP + GLCM + HOG + colour, from code/handcrafted_scores.py; no training involved).
  2. For neighbour pairs whose NOISY labels differ, the image with the higher loss (Task 1 noisy CE run,
     averaged over epochs 1..early stop) "loses" the pair and is flagged.
  3. Separately, the "memorization jump" of each image: how much its loss drops between the early-stop epoch
     and the final epoch (wrong labels are learned late, so their loss drops most).

Clean labels are used only to print how good each signal is (allowed Task 3 check).

    python code/pair_scores.py
"""
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from handcrafted_scores import prep  # noqa: E402


def softmax(z):
    z = z.astype(np.float32)
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def main():
    labels = pd.read_csv(os.path.join(CV, "data", "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(CV, "data", "train_idx.txt"), dtype=np.int64)
    y = labels.noisy_label.values[tr]
    clean = labels.clean_label.values[tr]                                   # measurement only
    wrong = y != clean
    k = int(wrong.sum())

    run = os.path.join(CV, "outputs", "runs", "t1_noisy")
    assert (np.load(os.path.join(run, "train_index.npy")) == tr).all()
    L = np.load(os.path.join(run, "train_logits.npy"), mmap_mode="r")
    log = pd.read_csv(os.path.join(run, "log.csv"))
    es = int(log.loc[log.val_acc_given.idxmax(), "epoch"])
    nll = lambda e: -np.log(np.clip(softmax(np.asarray(L[e - 1]))[np.arange(len(tr)), y], 1e-7, 1))  # noqa: E731
    loss_early = np.mean([nll(e) for e in range(1, es + 1)], 0)       # average loss before memorization
    jump = loss_early - nll(L.shape[0])                                # how much the loss fell by the end
    aum = np.load(os.path.join(CV, "outputs", "task3", "scores.npy"))[tr]

    B = dict(np.load(os.path.join(CV, "outputs", "task3", "handcrafted_features.npz")))
    feat = prep(np.concatenate([(B[p] - B[p].mean(0)) / (B[p].std(0) + 1e-6)
                                for p in ["color", "lbp", "glcm", "hog"]], 1), 64)
    f = torch.from_numpy(feat).cuda().half()
    nn_k = 10
    nbrs = np.zeros((len(tr), nn_k), np.int64)
    for i in range(0, len(tr), 2000):
        sim = f[i:i + 2000] @ f.T
        sim[torch.arange(len(sim)), torch.arange(i, i + len(sim))] = -2
        nbrs[i:i + 2000] = sim.topk(nn_k, 1).indices.cpu().numpy()

    print(f"[measurement] hand-crafted nearest neighbours that share the TRUE class: "
          f"{(clean[nbrs[:, 0]] == clean).mean():.1%} (1st), {(clean[nbrs] == clean[:, None]).mean():.1%} (top {nn_k})")

    # pair duel: among neighbours with a different noisy label, the fraction this image loses (higher loss)
    diff = y[nbrs] != y[:, None]
    lose = (loss_early[:, None] > loss_early[nbrs]) & diff
    duel = np.where(diff.any(1), lose.sum(1) / np.maximum(diff.sum(1), 1), 0.0)
    pair_flag = lose[:, 0]                                                  # 1st neighbour only, hard flag

    r = lambda s: pd.Series(s).rank(pct=True).values  # noqa: E731
    sig = {"AUM (submitted)": aum, "early loss (epochs 1..es)": loss_early,
           "pair duel (10 hand-crafted neighbours)": duel, "memorization jump": jump,
           "AUM + pair duel (rank avg)": (r(aum) + r(duel)) / 2,
           "AUM + memorization jump (rank avg)": (r(aum) + r(jump)) / 2}
    rows = [dict(signal=n, auroc=roc_auc_score(wrong, s), p_at_k=wrong[np.argsort(-s, kind="stable")[:k]].mean())
            for n, s in sig.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    flagged = pair_flag & diff[:, 0]
    print(f"\n[measurement] pairs with different labels (1st neighbour): {diff[:, 0].mean():.1%} of images; "
          f"the higher-loss member is truly mislabeled {wrong[flagged].mean():.1%} of the time "
          f"(base rate {wrong.mean():.1%})")


if __name__ == "__main__":
    main()
