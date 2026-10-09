"""Task 4 method: two-network ELR ("ELR+"). Two ResNet-18s (the 2-network budget), 60 epochs each, same
optimizer / LR schedule / batch size / crop+flip augmentation as the baseline.

  * each net k keeps a weight-averaged copy (mean-teacher style, momentum ramped up to --ema)
  * ELR targets for net k: running average (momentum --beta) of the OTHER net's weight-averaged
    predictions on the un-mixed images
  * loss for net k on a mixup batch: CE(mixed noisy labels) + lambda * log(1 - <p, mixed targets>)
  * predictions = average softmax of the two weight-averaged nets; best epoch by validation accuracy on the
    training labels (noisy or synthetic)

    python code/elrplus.py --name t4_elrplus --label noisy
"""
import argparse
import copy
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--elr_lambda", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--ema", type=float, default=0.997)
    ap.add_argument("--mixup", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    if a.epochs > MAX_EPOCHS:
        ap.error(f"budget is at most {MAX_EPOCHS} epochs")
    a.train_idx_file, a.cv_folds, a.targets_file, a.label_smoothing = None, 0, None, 0.0
    return a


@torch.no_grad()
def ema_update(ema, net, m):
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.mul_(m).add_(p.detach(), alpha=1 - m)
    for be, b in zip(ema.buffers(), net.buffers()):
        be.copy_(b)


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
        json.dump(vars(args) | {"method": "elr_plus", "gpu": torch.cuda.get_device_name(0)}, f, indent=2)

    D = load_data(args, device)
    n = len(D["y_train"])
    y1h = F.one_hot(D["y_train"], 10).float()
    correct_mask = D["y_train"] == D["train_clean"]  # measurement only
    norm = lambda x: ((x - D["mean"]) / D["std"]).contiguous(memory_format=torch.channels_last)  # noqa: E731
    B = args.batch_size
    steps = (n + B - 1) // B
    if args.limit_batches:
        steps = min(steps, args.limit_batches)

    nets, emas, opts, scheds = [], [], [], []
    for k in range(2):
        torch.manual_seed(args.seed * 2 + k)
        net = ResNet18(10).to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                              nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15, anneal_strategy="cos",
            div_factor=10, final_div_factor=1000, cycle_momentum=False)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        nets.append(net); emas.append(ema); opts.append(opt); scheds.append(sched)
    scalers = [torch.amp.GradScaler(enabled=amp) for _ in range(2)]
    targets = [torch.zeros(n, 10, device=device) for _ in range(2)]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    step_count = [0, 0]

    E = args.epochs
    test_logits = np.zeros((E, len(D["x_test"]), 10), np.float16)
    val_logits = np.zeros((E, len(D["x_val"]), 10), np.float16)
    log = []
    for ep in range(1, E + 1):
        t0 = time.time()
        for k in range(2):
            net, other_ema = nets[k], emas[1 - k]
            net.train(); other_ema.train()  # BN of the averaged copy uses batch stats while building targets
            perm = torch.randperm(n, device=device, generator=gen)
            for s in range(steps):
                bi = perm[s * B:(s + 1) * B]
                x = norm(augment(D["x_train"][bi]))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    pt = other_ema(x).float().softmax(1).clamp(1e-4, 1 - 1e-4)
                targets[k][bi] = args.beta * targets[k][bi] + (1 - args.beta) * pt / pt.sum(1, keepdim=True)
                lam = float(np.random.beta(args.mixup, args.mixup))
                mp = torch.randperm(len(bi), device=device)
                xm = (lam * x + (1 - lam) * x[mp]).contiguous(memory_format=torch.channels_last)
                ym = lam * y1h[bi] + (1 - lam) * y1h[bi][mp]
                tm = lam * targets[k][bi] + (1 - lam) * targets[k][bi][mp]
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = net(xm).float()
                p = out.softmax(1).clamp(1e-4, 1 - 1e-4)
                loss = -(ym * F.log_softmax(out, 1)).sum(1).mean() \
                    + args.elr_lambda * torch.log(1 - (tm * p).sum(1)).mean()
                opts[k].zero_grad(set_to_none=True)
                scalers[k].scale(loss).backward()
                scalers[k].step(opts[k]); scalers[k].update(); scheds[k].step()
                step_count[k] += 1
                ema_update(emas[k], net, min(args.ema, 1 - 1 / (step_count[k] + 1)))

        # ---- evaluation: averaged softmax of the two weight-averaged nets
        pv = sum(predict(m, D["x_val"], D["mean"], D["std"], amp).softmax(1) for m in emas) / 2
        ptest = sum(predict(m, D["x_test"], D["mean"], D["std"], amp).softmax(1) for m in emas) / 2
        ptr = sum(predict(m, D["x_train"], D["mean"], D["std"], amp).softmax(1) for m in emas) / 2
        lv, lt = pv.clamp_min(1e-7).log(), ptest.clamp_min(1e-7).log()
        val_logits[ep - 1] = lv.cpu().numpy()
        test_logits[ep - 1] = lt.cpu().numpy()
        row = dict(epoch=ep, lr=opts[0].param_groups[0]["lr"],
                   train_acc_given=acc(ptr, D["y_train"]),
                   train_acc_correct=acc(ptr, D["y_train"], correct_mask),
                   train_acc_mislabeled=acc(ptr, D["y_train"], ~correct_mask),
                   train_mislabeled_pred_clean=acc(ptr, D["train_clean"], ~correct_mask),
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
