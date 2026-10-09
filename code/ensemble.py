"""Average the softmax outputs of two runs of the same method (the 2-network budget) epoch by epoch,
then pick the best epoch with validation accuracy on the training label column (noisy or synthetic).

    python code/ensemble.py --runs t4_cvrelabel t4_cvrelabel_s1 --name t4_cvrelabel_ens

Writes outputs/runs/<name>/ with log.csv, pred_final.csv, pred_best.csv, test_logits.npy, summary.json,
in the same format as train.py, so the rest of the pipeline treats it like any other run.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
COLS = {"clean": "clean_label", "noisy": "noisy_label", "synthetic": "synthetic_label"}


def softmax(z):
    z = z.astype(np.float32)
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs_dir", default=os.path.join(CV, "outputs", "runs"))
    ap.add_argument("--data_dir", default=os.path.join(CV, "data"))
    a = ap.parse_args()

    cfgs = [json.load(open(os.path.join(a.runs_dir, r, "config.json"))) for r in a.runs]
    label = cfgs[0]["label"]
    assert all(c["label"] == label for c in cfgs), "runs were trained on different label columns"
    labels = pd.read_csv(os.path.join(a.data_dir, "labels.csv")).set_index("index").sort_index()
    val_idx = np.loadtxt(os.path.join(a.data_dir, "val_idx.txt"), dtype=np.int64)
    yv = {k: labels.loc[val_idx, c].values for k, c in COLS.items()}

    pv = sum(softmax(np.load(os.path.join(a.runs_dir, r, "val_logits.npy"))) for r in a.runs) / 2
    pt = sum(softmax(np.load(os.path.join(a.runs_dir, r, "test_logits.npy"))) for r in a.runs) / 2
    rows = []
    for e in range(pv.shape[0]):
        pred = pv[e].argmax(1)
        rows.append(dict(epoch=e + 1, val_acc_given=(pred == yv[label]).mean(),
                         val_acc_noisy=(pred == yv["noisy"]).mean(), val_acc_clean=(pred == yv["clean"]).mean(),
                         val_acc_synthetic=(pred == yv["synthetic"]).mean()))
    df = pd.DataFrame(rows)
    best = int(df.loc[df.val_acc_given.idxmax(), "epoch"])

    out = os.path.join(a.runs_dir, a.name)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "log.csv"), index=False)
    for fname, e in [("pred_final.csv", len(df)), ("pred_best.csv", best)]:
        pd.DataFrame({"id": np.arange(pt.shape[1]), "pred": pt[e - 1].argmax(1)}).to_csv(os.path.join(out, fname),
                                                                                     index=False)
    np.save(os.path.join(out, "test_logits.npy"), np.log(np.clip(pt, 1e-7, 1)).astype(np.float16))
    np.save(os.path.join(out, "val_logits.npy"), np.log(np.clip(pv, 1e-7, 1)).astype(np.float16))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(label=label, ensemble_of=a.runs), f, indent=2)
    summary = dict(best_epoch=best, val_acc_given_best=float(df.val_acc_given.max()),
                   val_acc_clean_at_best=float(df.val_acc_clean.iloc[best - 1]),
                   val_acc_given_final=float(df.val_acc_given.iloc[-1]))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(a.name, summary)


if __name__ == "__main__":
    main()
