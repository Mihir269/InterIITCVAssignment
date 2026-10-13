"""Test predictions from released weights (models/*.pt, final-epoch weights stored in fp16).

    python code/predict.py --weights models/t1_noisy.pt --out t1_noisy_ce.csv
    python code/predict.py --weights models/t5_elr_mixup_synth_seed0.pt models/t5_elr_mixup_synth_seed1.pt --out t5_best_synth.csv

With two weight files the softmax outputs are averaged (the 2-network methods). Inputs are normalised with the
constants the network was trained with (--norm, see NORM).
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


# Per-channel (mean, std) used to normalise inputs during training. Most runs computed them with a float32 reduction
# over all 45k training images, which loses precision (mean 0.364 for every channel instead of 0.49 / 0.48 / 0.45);
# harmless, because training and inference use the same constants. Later runs (the distribution-alignment pair)
# compute them exactly in float64 chunks. Inference must use the constants the network was trained with.
NORM = {
    "legacy": ([0.36408889293670654] * 3, [0.25979742407798767, 0.252744197845459, 0.2554599344730377]),
    "exact": ([0.49152732, 0.48229828, 0.44667101], [0.24700601, 0.24345457, 0.26157331]),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--norm", choices=list(NORM), default="legacy",
                    help="input normalisation the network was trained with: 'legacy' for every released network "
                         "except the distribution-alignment pair, which uses 'exact' (see NORM above)")
    ap.add_argument("--rebalance", default=None,
                    help="JSON with 'tau' and 'predicted_share_train' (written by code/rebalance.py): shift the "
                         "log-probabilities by -tau * log(share / 0.1) before the argmax")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = torch.tensor(NORM[a.norm][0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORM[a.norm][1], device=device).view(1, 3, 1, 1)
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
    score = (prob / len(a.weights)).clamp_min(1e-7).log()
    if a.rebalance:
        import json
        cfg = json.load(open(a.rebalance))
        score = score - cfg["tau"] * torch.log(torch.tensor(cfg["predicted_share_train"]) / 0.1)
    pd.DataFrame({"id": np.arange(len(test)), "pred": score.argmax(1).numpy()}).to_csv(a.out, index=False)
    print(f"wrote {a.out} from {len(a.weights)} network(s)")


if __name__ == "__main__":
    main()
