"""Test predictions from released weights (models/*.pt, final-epoch weights stored in fp16).

    python code/predict.py --weights models/t1_noisy.pt --out t1_noisy_ce.csv
    python code/predict.py --weights models/t5_elr_mixup_synth_seed0.pt models/t5_elr_mixup_synth_seed1.pt --out t5_best_synth.csv

With two weight files the softmax outputs are averaged (the 2-network methods). Inputs are normalised with the
mean / std of the 45k training images, exactly as in train.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import ResNet18  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_idx = np.loadtxt(os.path.join(a.data_dir, "train_idx.txt"), dtype=np.int64)
    tr = np.load(os.path.join(a.data_dir, "train_images.npy"), mmap_mode="r")[tr_idx].astype(np.float32) / 255
    mean = torch.tensor(tr.mean((0, 1, 2)), device=device).view(1, 3, 1, 1)
    std = torch.tensor(tr.std((0, 1, 2)), device=device).view(1, 3, 1, 1)
    del tr
    test = torch.from_numpy(np.load(os.path.join(a.data_dir, "test_images.npy"))).permute(0, 3, 1, 2)

    prob = 0
    for w in a.weights:
        net = ResNet18(10).to(device)
        sd = torch.load(w, map_location=device)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        net.eval()
        out = []
        for i in range(0, len(test), 1000):
            x = (test[i:i + 1000].to(device).float() / 255 - mean) / std
            out.append(net(x).softmax(1).cpu())
        prob = prob + torch.cat(out)
    pd.DataFrame({"id": np.arange(len(test)), "pred": prob.argmax(1).numpy()}).to_csv(a.out, index=False)
    print(f"wrote {a.out} from {len(a.weights)} network(s)")


if __name__ == "__main__":
    main()
