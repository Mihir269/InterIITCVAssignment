"""Task 1 (analysis part): noise rate and noisy-vs-clean confusion matrix.

Clean labels are used here only to *measure* the noise, which the rules allow.

    python code/analyze_noise.py            # run from the CV/ folder
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(data_dir):
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv")).set_index("index").sort_index()
    train_idx = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=int)
    val_idx = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=int)
    with open(os.path.join(data_dir, "classes.txt")) as f:
        classes = [c.strip() for c in f if c.strip()]
    return labels, train_idx, val_idx, classes


def confusion(clean, noisy, k=10):
    """cm[i, j] = number of images whose true class is i and whose given label is j."""
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (clean, noisy), 1)
    return cm


def summarize(name, clean, noisy, classes, out_dir, top=10):
    cm = confusion(clean, noisy)
    n = len(clean)
    wrong = (clean != noisy).sum()
    print(f"\n=== {name} ({n} images) ===")
    print(f"noise rate: {wrong / n:.4f}  ({wrong} mislabeled)")

    # per-class noise: of the images truly in class i, how many got a wrong label
    print("\nper true class: fraction mislabeled, and the wrong label it most often gets")
    for i, c in enumerate(classes):
        row = cm[i].copy()
        row_wrong = row.sum() - row[i]
        row[i] = -1
        j = row.argmax()
        print(f"  {c:11s} {row_wrong / cm[i].sum():6.1%}   most often called '{classes[j]}' "
              f"({cm[i, j] / cm[i].sum():.1%})")

    # directed pairs: true i -> labeled j
    off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    off.sort(reverse=True)
    print(f"\ntop {top} directed confusions (true -> given label), share of all errors:")
    cum = 0
    for cnt, i, j in off[:top]:
        cum += cnt
        print(f"  {classes[i]:>10s} -> {classes[j]:<10s} {cnt:5d}  {cnt / wrong:6.1%}  (cum {cum / wrong:6.1%})")

    # undirected pairs: {i, j} in either direction
    pairs = [(cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)]
    pairs.sort(reverse=True)
    print(f"\ntop {top} class pairs (both directions combined):")
    cum = 0
    for cnt, i, j in pairs[:top]:
        cum += cnt
        print(f"  {classes[i]:>10s} <-> {classes[j]:<10s} {cnt:5d}  {cnt / wrong:6.1%}  (cum {cum / wrong:6.1%})")

    # how the errors are spread out: a uniform spread over the 90 off-diagonal cells is ~1.1% each
    shares = np.sort(np.array([c for c, _, _ in off]) / wrong)[::-1]
    print(f"\nconcentration: top 5 of 90 directed cells hold {shares[:5].sum():.1%} of errors, "
          f"top 10 hold {shares[:10].sum():.1%} (uniform would be {5/90:.1%} / {10/90:.1%})")

    tag = name.lower().replace(" ", "_")
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"label_{c}" for c in classes]) \
        .to_csv(os.path.join(out_dir, f"confusion_{tag}.csv"))
    return cm, wrong / n


def plot(cms, titles, classes, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(7 * len(cms), 6))
    for ax, cm, title in zip(np.atleast_1d(axes), cms, titles):
        norm = cm / cm.sum(1, keepdims=True)  # row-normalized: P(given label | true class)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(10), classes, rotation=45, ha="right")
        ax.set_yticks(range(10), classes)
        ax.set_xlabel("given (noisy) label")
        ax.set_ylabel("true (clean) label")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "outputs", "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels, train_idx, val_idx, classes = load(args.data_dir)
    tr = labels.loc[train_idx]
    va = labels.loc[val_idx]

    cm_h, _ = summarize("train human", tr.clean_label.values, tr.noisy_label.values, classes, args.out_dir)
    summarize("val human", va.clean_label.values, va.noisy_label.values, classes, args.out_dir, top=5)
    cm_s, _ = summarize("train synthetic", tr.clean_label.values, tr.synthetic_label.values, classes,
                        args.out_dir, top=5)

    plot([cm_h, cm_s], ["Human noise (noisy_label), train", "Synthetic noise (synthetic_label), train"],
         classes, os.path.join(args.out_dir, "confusion_train.png"))


if __name__ == "__main__":
    main()
