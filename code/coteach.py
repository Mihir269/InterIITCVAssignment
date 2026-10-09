"""Task 4 method: Co-teaching (Han et al., NeurIPS 2018). Two ResNet-18s (the 2-network budget), same
recipe as the baseline. In every batch each network ranks samples by its own loss and passes its
small-loss fraction to the OTHER network for the update. Predictions = average softmax of the two nets.

    python code/coteach.py --name t4_coteach --forget_rate 0.35

--forget_rate must be set WITHOUT clean labels, e.g. from the Task 3 cross-validated models
(fraction of training images whose held-out prediction disagrees with the noisy label).

Outputs match train.py: log.csv, pred_final.csv, pred_best.csv (best noisy-val epoch), *_logits.npy.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import ResNet18  # noqa: E402
from train import MAX_EPOCHS, acc, augment, load_data, predict, write_preds  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "outputs", "runs"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", choices=["noisy", "synthetic"], default="noisy")
    ap.add_argument("--forget_rate", type=float, required=True)
    ap.add_argument("--num_gradual", type=int, default=10, help="epochs to ramp the forget rate up")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    if a.epochs > MAX_EPOCHS:
        ap.error(f"budget is at most {MAX_EPOCHS} epochs")
    # fields load_data expects
    a.train_idx_file, a.cv_folds, a.targets_file, a.label_smoothing = None, 0, None, 0.0
    return a


def main():
    args = get_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    amp = not args.no_amp
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"method": "coteaching", "gpu": torch.cuda.get_device_name(0)}, f, indent=2)

    D = load_data(args, device)
    n = len(D["y_train"])
    correct_mask = D["y_train"] == D["train_clean"]  # measurement only

    nets, opts, scheds = [], [], []
    steps = (n + args.batch_size - 1) // args.batch_size
    if args.limit_batches:
        steps = min(steps, args.limit_batches)
    for k in range(2):
        torch.manual_seed(args.seed + k)  # different init per net
        net = ResNet18(10).to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15, anneal_strategy="cos",
            div_factor=10, final_div_factor=1000, cycle_momentum=False)
        nets.append(net); opts.append(opt); scheds.append(sched)
    scalers = [torch.amp.GradScaler(enabled=amp) for _ in range(2)]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    E = args.epochs
    test_logits = np.zeros((E, len(D["x_test"]), 10), np.float16)
    val_logits = np.zeros((E, len(D["x_val"]), 10), np.float16)
    log = []
    for ep in range(1, E + 1):
        t0 = time.time()
        rate = args.forget_rate * min(ep / args.num_gradual, 1.0)
        for net in nets:
            net.train()
        perm = torch.randperm(n, device=device, generator=gen)
        tot, seen, kept_clean = 0.0, 0, 0.0
        for s in range(steps):
            bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = ((augment(D["x_train"][bi]) - D["mean"]) / D["std"]).contiguous(memory_format=torch.channels_last)
            yb = D["y_train"][bi]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outs = [net(xb) for net in nets]
            losses = [F.cross_entropy(o.float(), yb, reduction="none") for o in outs]
            keep = max(1, int(round(len(bi) * (1 - rate))))
            sel = [torch.argsort(l.detach())[:keep] for l in losses]  # each net's small-loss samples
            for k in range(2):  # net k learns from the samples chosen by the other net
                loss = losses[k][sel[1 - k]].mean()
                opts[k].zero_grad(set_to_none=True)
                scalers[k].scale(loss).backward()
                scalers[k].step(opts[k]); scalers[k].update(); scheds[k].step()
                tot += loss.item() * len(bi) / 2
            kept_clean += correct_mask[bi][sel[0]].float().mean().item()  # measurement only
            seen += len(bi)

        pv = sum(predict(net, D["x_val"], D["mean"], D["std"], amp).softmax(1) for net in nets) / 2
        pt = sum(predict(net, D["x_test"], D["mean"], D["std"], amp).softmax(1) for net in nets) / 2
        ptr = sum(predict(net, D["x_train"], D["mean"], D["std"], amp).softmax(1) for net in nets) / 2
        lv, lt = pv.clamp_min(1e-7).log(), pt.clamp_min(1e-7).log()
        val_logits[ep - 1] = lv.cpu().numpy()
        test_logits[ep - 1] = lt.cpu().numpy()
        row = dict(epoch=ep, lr=opts[0].param_groups[0]["lr"], forget_rate=rate, train_loss=tot / seen,
                   selected_precision=kept_clean / steps,  # measurement only: share of kept samples that are clean
                   train_acc_given=acc(ptr, D["y_train"]),
                   train_acc_correct=acc(ptr, D["y_train"], correct_mask),
                   train_acc_mislabeled=acc(ptr, D["y_train"], ~correct_mask),
                   val_acc_given=acc(lv, D["y_val"]), val_acc_noisy=acc(lv, D["val_noisy"]),
                   val_acc_clean=acc(lv, D["val_clean"]), val_acc_synthetic=acc(lv, D["val_synth"]),
                   time_s=time.time() - t0)
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        pd.DataFrame(log).to_csv(os.path.join(run_dir, "log.csv"), index=False)

    df = pd.DataFrame(log)
    best = int(df.loc[df.val_acc_given.idxmax(), "epoch"])
    write_preds(os.path.join(run_dir, "pred_final.csv"), test_logits[-1])
    write_preds(os.path.join(run_dir, "pred_best.csv"), test_logits[best - 1])
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "val_logits.npy"), val_logits)
    summary = dict(best_epoch=best, val_acc_given_best=df.val_acc_given.max(),
                   val_acc_given_final=df.val_acc_given.iloc[-1], val_acc_clean_final=df.val_acc_clean.iloc[-1],
                   total_time_min=df.time_s.sum() / 60)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
