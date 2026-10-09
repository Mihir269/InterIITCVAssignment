"""Task 3 (extra): feature-space outlier scores for mislabeled images.

Each training image is described by the 512-d penultimate-layer feature of a trained ResNet-18 (eval mode,
no augmentation, L2-normalised). Images are grouped by their NOISY label and scored (higher = label more
likely wrong):

  own_centroid     cosine distance to the mean feature of its own (noisy) class
  centroid_margin  k-means with --k clusters inside every class; distance to the nearest cluster of its own
                   class minus distance to the nearest cluster of any other class
  knn              share of its --nn nearest neighbours (cosine) that carry a different noisy label

Feature sources:
  --source <run>   one model's features for all 45k training images
  --source cv      the 5 Task 3 fold models: every image is described by the fold model that never trained on
                   it, and neighbours / clusters are formed among that fold's held-out images only

Clean labels are used only to print AUROC / precision@k (the allowed Task 3 check).

    python code/feature_scores.py --source s_ce_mix1
    python code/feature_scores.py --source cv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from model import ResNet18  # noqa: E402

RUNS = os.path.join(CV, "outputs", "runs")


@torch.no_grad()
def features(run, imgs, mean, std, device="cuda", bs=500):
    net = ResNet18(10).to(device)
    net.load_state_dict(torch.load(os.path.join(RUNS, run, "model_final.pt"), map_location=device))
    net.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            h = F.relu(net.bn1(net.conv1(x)))
            h = net.layer4(net.layer3(net.layer2(net.layer1(h))))
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        out.append(F.normalize(h.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def scores_for(feat, y, k_clusters, nn, device="cuda"):
    n = len(y)
    s = {}
    # own centroid
    cent = np.stack([feat[y == c].mean(0) for c in range(10)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    s["own_centroid"] = 1 - (feat * cent[y]).sum(1)
    # per-class k-means, margin to the nearest other-class cluster
    centers, owner = [], []
    for c in range(10):
        km = KMeans(n_clusters=k_clusters, n_init=3, random_state=0).fit(feat[y == c])
        cc = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        centers.append(cc); owner += [c] * k_clusters
    centers, owner = np.concatenate(centers), np.array(owner)
    d = 1 - feat @ centers.T                                   # (n, 10k)
    own = np.where(owner[None, :] == y[:, None], d, np.inf).min(1)
    other = np.where(owner[None, :] != y[:, None], d, np.inf).min(1)
    s["centroid_margin"] = own - other
    # kNN label disagreement (cosine), computed in chunks on the GPU
    f = torch.from_numpy(feat).to(device).half()
    yt = torch.from_numpy(y).to(device)
    dis = np.zeros(n)
    for i in range(0, n, 2000):
        sim = f[i:i + 2000] @ f.T
        sim[torch.arange(len(sim)), torch.arange(i, i + len(sim))] = -2  # exclude self
        nb = sim.topk(nn, dim=1).indices
        dis[i:i + 2000] = (yt[nb] != yt[i:i + 2000, None]).float().mean(1).cpu().numpy()
    s["knn"] = dis
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="run name, or 'cv' for the 5 Task 3 fold models")
    ap.add_argument("--k", type=int, default=5, help="k-means clusters per class")
    ap.add_argument("--nn", type=int, default=50, help="neighbours for the kNN score")
    a = ap.parse_args()

    d = os.path.join(CV, "data")
    labels = pd.read_csv(os.path.join(d, "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(d, "train_idx.txt"), dtype=np.int64)
    imgs = np.load(os.path.join(d, "train_images.npy"), mmap_mode="r")
    t = imgs[tr].astype(np.float32) / 255
    mean = torch.tensor(t.mean((0, 1, 2)), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(t.std((0, 1, 2)), device="cuda").view(1, 3, 1, 1)
    del t
    noisy = labels.noisy_label.values

    full = {}
    if a.source == "cv":
        for f in range(5):
            run = f"t3_cv{f}"
            idx = np.load(os.path.join(RUNS, run, "holdout_index.npy"))
            feat = features(run, np.asarray(imgs[idx]), mean, std)
            for name, v in scores_for(feat, noisy[idx], a.k, a.nn).items():
                full.setdefault(name, np.full(len(labels), np.nan))[idx] = v
            print(f"fold {f}: {len(idx)} held-out images")
    else:
        feat = features(a.source, np.asarray(imgs[tr]), mean, std)
        for name, v in scores_for(feat, noisy[tr], a.k, a.nn).items():
            full.setdefault(name, np.full(len(labels), np.nan))[tr] = v

    out = os.path.join(CV, "outputs", "task3")
    os.makedirs(out, exist_ok=True)
    wrong = (labels.noisy_label.values != labels.clean_label.values)[tr]  # measurement only
    kk = int(wrong.sum())
    aum = np.load(os.path.join(out, "scores.npy"))[tr] if os.path.exists(os.path.join(out, "scores.npy")) else None
    rows = []
    for name, v in full.items():
        np.save(os.path.join(out, f"feat_{a.source}_{name}.npy"), v)
        st = v[tr]
        rows.append(dict(signal=f"{a.source}:{name}", auroc=roc_auc_score(wrong, st),
                         precision_at_k=wrong[np.argsort(-st, kind="stable")[:kk]].mean()))
        if aum is not None:
            r = (pd.Series(st).rank(pct=True).values + pd.Series(aum).rank(pct=True).values) / 2
            rows.append(dict(signal=f"  + AUM rank-avg", auroc=roc_auc_score(wrong, r),
                             precision_at_k=wrong[np.argsort(-r, kind="stable")[:kk]].mean()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
