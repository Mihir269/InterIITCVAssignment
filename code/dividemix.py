"""Task 4 method: DivideMix-style training (Li et al., ICLR 2020), simplified to the bootcamp budget.

Two ResNet-18s (the 2-network budget), 60 epochs each, same optimizer / LR schedule / batch size /
crop+flip augmentation as the baseline. Labels used: the training label column only (noisy or synthetic).

  * warm-up (default 10 epochs): plain cross-entropy for each net, as in the baseline.
  * every later epoch, for each net k:
      - co-divide: fit a 2-component Gaussian mixture to the per-sample loss of the OTHER net (eval mode,
        no augmentation); w_i = posterior of the low-loss component = probability the label is clean.
        Samples with w_i > p_threshold form the labeled set, the rest the unlabeled set.
      - label refinement (labeled):  t = sharpen(w * onehot(y) + (1 - w) * mean_aug p_k(x))
      - label co-guessing (unlabeled): t = sharpen(mean over 2 augs and both nets of p(x))
      - MixMatch: mixup (lambda ~ Beta(4, 4), lambda >= 0.5) over labeled + unlabeled batches; loss =
        soft CE on the labeled part + lambda_u * MSE on the unlabeled part + uniform-prior penalty.
  * predictions = average softmax of the two nets; best epoch by validation accuracy on the training labels.

    python code/dividemix.py --name t4_dividemix --label noisy
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

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
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--p_threshold", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.5, help="sharpening temperature")
    ap.add_argument("--alpha", type=float, default=4.0, help="mixup Beta(alpha, alpha)")
    ap.add_argument("--lambda_u", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_batches", type=int, default=0)
    a = ap.parse_args()
    if a.epochs > MAX_EPOCHS:
        ap.error(f"budget is at most {MAX_EPOCHS} epochs")
    a.train_idx_file, a.cv_folds, a.targets_file, a.label_smoothing = None, 0, None, 0.0
    return a


def lr_at(progress, max_lr):
    """Same shape as the baseline's OneCycleLR: cosine warm-up 0.1x -> 1x over 15%, cosine decay to 1e-4x."""
    lo, end, pct = max_lr / 10, max_lr / 10 / 1000, 0.15
    if progress < pct:
        return lo + (max_lr - lo) * (1 - math.cos(math.pi * progress / pct)) / 2
    q = (progress - pct) / (1 - pct)
    return end + (max_lr - end) * (1 + math.cos(math.pi * min(q, 1.0))) / 2


def sharpen(p, T):
    p = p ** (1 / T)
    return p / p.sum(1, keepdim=True)


@torch.no_grad()
def per_sample_loss(net, D, amp):
    logits = predict(net, D["x_train"], D["mean"], D["std"], amp)
    return F.cross_entropy(logits, D["y_train"], reduction="none"), logits


def gmm_clean_prob(loss):
    l = loss.cpu().numpy().astype(np.float64)
    finite = np.isfinite(l)
    l = np.where(finite, l, l[finite].max() if finite.any() else 0.0)  # fp16 overflow guard
    l = (l - l.min()) / (l.max() - l.min() + 1e-12)
    g = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=0)
    g.fit(l[:, None])
    post = g.predict_proba(l[:, None])
    return post[:, g.means_.argmin()]


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
        json.dump(vars(args) | {"method": "dividemix", "gpu": torch.cuda.get_device_name(0)}, f, indent=2)

    D = load_data(args, device)
    n = len(D["y_train"])
    y1h = F.one_hot(D["y_train"], 10).float()
    correct_mask = D["y_train"] == D["train_clean"]  # measurement only
    norm = lambda x: ((x - D["mean"]) / D["std"]).contiguous(memory_format=torch.channels_last)  # noqa: E731
    B = args.batch_size

    nets, opts = [], []
    for k in range(2):
        torch.manual_seed(args.seed * 2 + k)
        nets.append(ResNet18(10).to(device).to(memory_format=torch.channels_last))
        opts.append(torch.optim.SGD(nets[k].parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.wd, nesterov=True))
    scalers = [torch.amp.GradScaler(enabled=amp) for _ in range(2)]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    E = args.epochs

    def set_lr(k, progress):
        for g in opts[k].param_groups:
            g["lr"] = lr_at(progress, args.lr)

    def step(k, loss):
        opts[k].zero_grad(set_to_none=True)
        scalers[k].scale(loss).backward()
        scalers[k].step(opts[k])
        scalers[k].update()

    test_logits = np.zeros((E, len(D["x_test"]), 10), np.float16)
    val_logits = np.zeros((E, len(D["x_val"]), 10), np.float16)
    log = []
    full_steps = (n + B - 1) // B
    for ep in range(1, E + 1):
        t0 = time.time()
        row = dict(epoch=ep)
        if ep <= args.warmup:  # ---- plain CE warm-up for both nets
            steps = min(full_steps, args.limit_batches or full_steps)
            for k in range(2):
                nets[k].train()
                perm = torch.randperm(n, device=device, generator=gen)
                for s in range(steps):
                    set_lr(k, (ep - 1 + s / steps) / E)
                    bi = perm[s * B:(s + 1) * B]
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        out = nets[k](norm(augment(D["x_train"][bi])))
                    step(k, F.cross_entropy(out.float(), D["y_train"][bi]))
        else:  # ---- co-divide + MixMatch
            w = [None, None]
            for k in range(2):
                loss_k, _ = per_sample_loss(nets[k], D, amp)
                w[k] = torch.from_numpy(gmm_clean_prob(loss_k)).float().to(device)
            for k in range(2):
                wk = w[1 - k]  # division made by the OTHER net
                lab = torch.nonzero(wk > args.p_threshold).squeeze(1)
                unl = torch.nonzero(wk <= args.p_threshold).squeeze(1)
                row[f"net{k}_labeled_frac"] = len(lab) / n
                row[f"net{k}_labeled_precision"] = correct_mask[lab].float().mean().item()  # measurement only
                if len(unl) == 0:
                    unl = lab
                steps = (len(lab) + B - 1) // B
                steps = min(steps, args.limit_batches or steps)
                net, other = nets[k], nets[1 - k]
                net.train(); other.train()
                lperm = lab[torch.randperm(len(lab), device=device, generator=gen)]
                for s in range(steps):
                    set_lr(k, (ep - 1 + s / steps) / E)
                    li = lperm[s * B:(s + 1) * B]
                    ui = unl[torch.randint(len(unl), (len(li),), device=device, generator=gen)]
                    xl1, xl2 = norm(augment(D["x_train"][li])), norm(augment(D["x_train"][li]))
                    xu1, xu2 = norm(augment(D["x_train"][ui])), norm(augment(D["x_train"][ui]))
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        pu = sum(m(x).float().softmax(1) for m in (net, other) for x in (xu1, xu2)) / 4
                        pl = (net(xl1).float().softmax(1) + net(xl2).float().softmax(1)) / 2
                    wl = wk[li][:, None]
                    tl = sharpen(wl * y1h[li] + (1 - wl) * pl, args.T)
                    tu = sharpen(pu, args.T)
                    x_all = torch.cat([xl1, xu1])
                    t_all = torch.cat([tl, tu])
                    lam = float(np.random.beta(args.alpha, args.alpha))
                    lam = max(lam, 1 - lam)
                    mp = torch.randperm(len(x_all), device=device)
                    xm = (lam * x_all + (1 - lam) * x_all[mp]).contiguous(memory_format=torch.channels_last)
                    tm = lam * t_all + (1 - lam) * t_all[mp]
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        out = net(xm).float()
                    nl = len(li)
                    lx = -(tm[:nl] * F.log_softmax(out[:nl], 1)).sum(1).mean()
                    lu = ((out[nl:].softmax(1) - tm[nl:]) ** 2).mean()
                    pm = out.softmax(1).mean(0)
                    penalty = (torch.full_like(pm, 0.1) * torch.log(0.1 / pm.clamp_min(1e-7))).sum()
                    step(k, lx + args.lambda_u * lu + penalty)

        # ---- evaluation: averaged softmax of the two nets
        pv = sum(predict(m, D["x_val"], D["mean"], D["std"], amp).softmax(1) for m in nets) / 2
        pt = sum(predict(m, D["x_test"], D["mean"], D["std"], amp).softmax(1) for m in nets) / 2
        ptr = sum(predict(m, D["x_train"], D["mean"], D["std"], amp).softmax(1) for m in nets) / 2
        lv, lt = pv.clamp_min(1e-7).log(), pt.clamp_min(1e-7).log()
        val_logits[ep - 1] = lv.cpu().numpy()
        test_logits[ep - 1] = lt.cpu().numpy()
        row |= dict(lr=opts[0].param_groups[0]["lr"],
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
