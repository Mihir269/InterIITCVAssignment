"""Measurement for Q5: are mislabeled images harder? (human noise vs random synthetic noise)

Difficulty of a validation image = confidence of the Task 1 CLEAN-label reference model (final epoch) in the
image's true class. That model never trained on validation images, so this reflects how visually ambiguous
the image is, independent of any annotator. "Hard" = lowest q% of that confidence.
Clean labels are used here only to measure (allowed); nothing is trained or selected with this.

    python code/noise_difficulty.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")


def main():
    labels = pd.read_csv(os.path.join(CV, "data", "labels.csv")).set_index("index").sort_index()
    vi = np.loadtxt(os.path.join(CV, "data", "val_idx.txt"), dtype=np.int64)
    z = np.load(os.path.join(CV, "outputs", "runs", "t1_clean", "val_logits.npy"))[-1].astype(np.float32)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    clean = labels.clean_label.values[vi]
    conf = p[np.arange(len(vi)), clean]  # confidence in the true class

    rows = []
    for col, name in [("noisy_label", "human"), ("synthetic_label", "random")]:
        wrong = labels[col].values[vi] != clean
        row = dict(noise=name, mislabeled=wrong.mean(),
                   conf_mislabeled=conf[wrong].mean(), conf_correct=conf[~wrong].mean())
        for q in [0.1, 0.2, 0.3, 0.5]:
            hard = conf < np.quantile(conf, q)
            row[f"rate_hardest_{int(q * 100)}%"] = wrong[hard].mean()
            row[f"rate_rest_{int(q * 100)}%"] = wrong[~hard].mean()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("noise").T
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
