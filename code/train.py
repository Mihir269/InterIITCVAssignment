"""Train the provided ResNet-18 on one label column and log everything Tasks 1, 2, 3 and 5 need.

    # Task 1 reference (clean labels) and Task 1/2 noisy baseline
    python code/train.py --label clean --name t1_clean
    python code/train.py --label noisy --name t1_noisy
    # Task 5 baseline (synthetic labels)
    python code/train.py --label synthetic --name t5_baseline_synth
    # quick smoke test
    python code/train.py --label noisy --name smoke --epochs 2 --limit_batches 20

Clean labels are used ONLY for training when --label clean, and otherwise ONLY for logged measurements
(train_acc_correct / train_acc_mislabeled / val_acc_clean). Early stopping ("best" epoch) uses the
validation accuracy measured against the *training label column* (noisy for noisy runs, synthetic for
synthetic runs), never the clean labels.

Outputs, in <out_dir>/<name>/:
    config.json        the exact settings
    log.csv            one row per epoch, all metrics
    dynamics.csv       the Task 2 columns (epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean)
    pred_final.csv     test predictions (id,pred) from the last epoch
    pred_best.csv      test predictions from the epoch with the best validation accuracy on the training labels
    test_logits.npy    (epochs, 10000, 10) float16
    val_logits.npy     (epochs, 5000, 10) float16
    train_logits.npy   (epochs, n_train, 10) float16, eval mode, no augmentation (for Task 3 signals)
    train_index.npy    the training indices in the row order of train_logits.npy
    holdout_logits.npy (epochs, n_holdout, 10) float16, only with --cv_folds (Task 3)
    holdout_index.npy  the held-out indices in the row order of holdout_logits.npy
    model_final.pt     final weights

Task 4 options (defaults = the plain CE baseline): --loss gce|sce, --label_smoothing, --mixup,
--targets_file (relabeled/soft targets), --train_idx_file (filtered subset).
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

MAX_EPOCHS = 60
LABEL_COLS = {"clean": "clean_label", "noisy": "noisy_label", "synthetic": "synthetic_label"}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "outputs", "runs"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", choices=list(LABEL_COLS), required=True)
    ap.add_argument("--train_idx_file", default=None,
                    help="file of training indices to use (default: data/train_idx.txt)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd",
                    help="sgd = the recipe used for every submitted run; adamw only for the optimizer comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--no_train_logits", action="store_true", help="skip the per-epoch train-set eval pass")
    ap.add_argument("--limit_batches", type=int, default=0, help="debug: stop each epoch after N batches")
    # --- Task 3: cross-validation. Train on all folds but one, save per-epoch logits on the held-out fold.
    ap.add_argument("--cv_folds", type=int, default=0)
    ap.add_argument("--cv_fold", type=int, default=0)
    ap.add_argument("--cv_seed", type=int, default=1234, help="fold split seed (same for every fold)")
    # --- Task 4: methods. Defaults reproduce the plain cross-entropy baseline.
    ap.add_argument("--loss", choices=["ce", "gce", "sce"], default="ce")
    ap.add_argument("--gce_q", type=float, default=0.7)
    ap.add_argument("--sce_alpha", type=float, default=0.1)
    ap.add_argument("--sce_beta", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a, a) parameter; 0 = off")
    ap.add_argument("--targets_file", default=None,
                    help=".npy (50000, 10) soft/hard training targets by global index, replacing the label column "
                         "(e.g. relabeled targets built from Task 3 models)")
    ap.add_argument("--forward_T", default=None,
                    help=".npy (10, 10) noise transition matrix T[true, given] ESTIMATED from noisy labels only "
                         "(code/estimate_T.py); enables forward loss correction (Patrini et al. 2017)")
    ap.add_argument("--elr_lambda", type=float, default=0.0,
                    help="ELR (early-learning regularization) weight; 0 = off")
    ap.add_argument("--elr_beta", type=float, default=0.7, help="ELR target running-average momentum")
    ap.add_argument("--sop", action="store_true",
                    help="SOP: per-sample sparse noise terms u_i^2*y - v_i^2*(1-y) absorb label errors")
    ap.add_argument("--sop_lr_u", type=float, default=10.0)
    ap.add_argument("--sop_lr_v", type=float, default=10.0)
    args = ap.parse_args()
    if args.sop and (args.mixup or args.loss != "ce" or args.targets_file or args.forward_T):
        ap.error("--sop is used on its own (plain one-hot labels, CE)")
    if args.epochs > MAX_EPOCHS:
        ap.error(f"budget is at most {MAX_EPOCHS} epochs")
    if args.cv_folds and not 0 <= args.cv_fold < args.cv_folds:
        ap.error("--cv_fold must be in [0, cv_folds)")
    return args


def loss_fn(logits, targets, args):
    """Loss against soft targets (rows sum to 1). With one-hot targets these are the usual CE / GCE / SCE."""
    logits = logits.float()
    logp = F.log_softmax(logits, 1)
    T = getattr(args, "T_mat", None)
    if T is not None:  # forward correction: the net predicts clean classes, T maps them to annotator labels
        logp = (logp.exp() @ T).clamp_min(1e-7).log()
    if args.loss == "ce":
        return -(targets * logp).sum(1).mean()
    p = logp.exp()
    if args.loss == "gce":  # Zhang & Sabuncu 2018: (1 - p_y^q) / q
        return (targets * (1 - p.clamp_min(1e-7) ** args.gce_q) / args.gce_q).sum(1).mean()
    # sce: Wang et al. 2019, alpha * CE + beta * reverse CE (log 0 clipped to log 1e-4)
    ce = -(targets * logp).sum(1).mean()
    rce = -(p * targets.clamp(1e-4, 1).log()).sum(1).mean()
    return args.sce_alpha * ce + args.sce_beta * rce


# ---------------------------------------------------------------- data
def load_data(args, device):
    d = args.data_dir
    labels = pd.read_csv(os.path.join(d, "labels.csv")).set_index("index").sort_index()
    full_train_idx = np.loadtxt(os.path.join(d, "train_idx.txt"), dtype=np.int64)
    train_idx = np.loadtxt(args.train_idx_file, dtype=np.int64) if args.train_idx_file else full_train_idx
    val_idx = np.loadtxt(os.path.join(d, "val_idx.txt"), dtype=np.int64)
    assert not set(train_idx) & set(val_idx), "train and val indices overlap"
    holdout_idx = np.zeros(0, np.int64)
    if args.cv_folds:
        fold_of = np.random.RandomState(args.cv_seed).permutation(len(train_idx)) % args.cv_folds
        holdout_idx = train_idx[fold_of == args.cv_fold]
        train_idx = train_idx[fold_of != args.cv_fold]

    imgs = np.load(os.path.join(d, "train_images.npy"))   # (50000, 32, 32, 3) uint8
    test = np.load(os.path.join(d, "test_images.npy"))    # (10000, 32, 32, 3) uint8

    def to_gpu(a):  # uint8 NCHW on the GPU; ~150 MB for all 50k training images
        return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)

    # normalization stats from all 45k training images (no labels involved), same for every run
    tr = imgs[full_train_idx].astype(np.float32) / 255
    mean = torch.tensor(tr.mean((0, 1, 2)), device=device).view(1, 3, 1, 1)
    std = torch.tensor(tr.std((0, 1, 2)), device=device).view(1, 3, 1, 1)
    del tr

    def col(name, idx):
        return torch.from_numpy(labels.loc[idx, name].values).to(device)

    given = LABEL_COLS[args.label]
    y_train = col(given, train_idx)
    if args.targets_file:  # e.g. relabeled / soft targets derived from noisy-label models
        t = np.load(args.targets_file).astype(np.float32)[train_idx]
        targets = torch.from_numpy(t / t.sum(1, keepdims=True)).to(device)
    else:
        targets = F.one_hot(y_train, 10).float()
    if args.label_smoothing:
        targets = (1 - args.label_smoothing) * targets + args.label_smoothing / 10
    return dict(
        x_train=to_gpu(imgs[train_idx]), x_val=to_gpu(imgs[val_idx]), x_test=to_gpu(test),
        x_holdout=to_gpu(imgs[holdout_idx]), holdout_idx=holdout_idx,
        train_idx=train_idx, val_idx=val_idx, mean=mean, std=std,
        targets=targets,                                     # what the loss sees
        y_train=y_train,                                     # the ONLY labels used for training
        y_val=col(given, val_idx),                           # the ONLY labels used for model selection
        # measurement only:
        train_clean=col("clean_label", train_idx),
        val_noisy=col("noisy_label", val_idx), val_clean=col("clean_label", val_idx),
        val_synth=col("synthetic_label", val_idx),
    )


def augment(x):
    """Random crop (zero-pad 4) + horizontal flip, on a uint8 GPU batch. Returns float in [0, 1]."""
    b = x.shape[0]
    x = F.pad(x.float() / 255, (4, 4, 4, 4))
    ar = torch.arange(32, device=x.device)
    rows = (torch.randint(0, 9, (b, 1), device=x.device) + ar)[:, None, :, None].expand(b, 3, 32, 40)
    cols = (torch.randint(0, 9, (b, 1), device=x.device) + ar)[:, None, None, :].expand(b, 3, 32, 32)
    x = x.gather(2, rows).gather(3, cols)
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(net, x, mean, std, amp, bs=1000):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = ((x[i:i + bs].float() / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(net(xb).float())
    return torch.cat(out)


def acc(logits, y, mask=None):
    hit = logits.argmax(1) == y
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        hit = hit[mask]
    return hit.float().mean().item()


def write_preds(path, logits):
    pd.DataFrame({"id": np.arange(len(logits)), "pred": logits.argmax(1)}).to_csv(path, index=False)


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
        json.dump(vars(args) | {"gpu": torch.cuda.get_device_name(0)}, f, indent=2)

    D = load_data(args, device)
    if args.forward_T:
        T = np.load(args.forward_T).astype(np.float32)
        args.T_mat = torch.from_numpy(T / T.sum(1, keepdims=True)).to(device)
    n_train = len(D["y_train"])
    correct_mask = D["y_train"] == D["train_clean"]  # measurement only
    print(f"[{args.name}] label={args.label}  n_train={n_train}  "
          f"given-label noise in train set={1 - correct_mask.float().mean().item():.4f}")

    net = ResNet18(10).to(device).to(memory_format=torch.channels_last)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.wd, nesterov=True)
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    if args.limit_batches:
        steps_per_epoch = min(steps_per_epoch, args.limit_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(  # cosine decay with a short warm-up, stepped per batch
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15,
        anneal_strategy="cos", div_factor=10, final_div_factor=1000, cycle_momentum=False)
    scaler = torch.amp.GradScaler(enabled=amp)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    E = args.epochs
    test_logits = np.zeros((E, len(D["x_test"]), 10), np.float16)
    val_logits = np.zeros((E, len(D["x_val"]), 10), np.float16)
    train_logits = None if args.no_train_logits else np.zeros((E, n_train, 10), np.float16)
    holdout_logits = np.zeros((E, len(D["holdout_idx"]), 10), np.float16) if args.cv_folds else None
    log = []
    if args.elr_lambda:  # running average of the network's own predictions, one row per training image
        elr_t = torch.zeros(n_train, 10, device=device)
    if args.sop:  # per-sample noise parameters, tiny initialisation, own plain-SGD optimizer (no decay)
        sop_u = torch.nn.Parameter(torch.randn(n_train, 10, device=device) * 1e-8)
        sop_v = torch.nn.Parameter(torch.randn(n_train, 10, device=device) * 1e-8)
        opt_uv = torch.optim.SGD([{"params": [sop_u], "lr": args.sop_lr_u},
                                  {"params": [sop_v], "lr": args.sop_lr_v}], momentum=0, weight_decay=0)

    for ep in range(1, E + 1):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n_train, device=device, generator=gen)
        tot_loss, seen = 0.0, 0
        for s in range(steps_per_epoch):
            bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = (augment(D["x_train"][bi]) - D["mean"]) / D["std"]
            x_plain = xb
            tb = D["targets"][bi]
            if args.mixup:
                lam = float(np.random.beta(args.mixup, args.mixup))
                mp = torch.randperm(len(bi), device=device)
                xb = lam * xb + (1 - lam) * xb[mp]
                tb = lam * tb + (1 - lam) * tb[mp]
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = net(xb)
            if args.sop:
                eps = 1e-4
                us = (sop_u[bi] ** 2 * tb).clamp(0, 1)
                vs = (sop_v[bi] ** 2 * (1 - tb)).clamp(0, 1)
                pred = (out.float().softmax(1) + us - vs.detach()).clamp_min(eps)
                pred = (pred / pred.sum(1, keepdim=True)).clamp(eps, 1.0)
                loss = -(tb * pred.log()).sum(1).mean()
                hard = F.one_hot(out.detach().float().argmax(1), 10).float()
                loss = loss + ((hard + us - vs - tb) ** 2).sum() / len(bi)
            else:
                loss = loss_fn(out, tb, args)
            if args.elr_lambda:
                p = out.float().softmax(1).clamp(1e-4, 1 - 1e-4)
                with torch.no_grad():
                    if args.mixup:  # targets are updated from the un-mixed images, then mixed like the labels
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            pc = net(x_plain.contiguous(memory_format=torch.channels_last)).float().softmax(1)
                    else:
                        pc = p.detach()
                    pc = pc.clamp(1e-4, 1 - 1e-4)
                    elr_t[bi] = args.elr_beta * elr_t[bi] + (1 - args.elr_beta) * pc / pc.sum(1, keepdim=True)
                    t_b = elr_t[bi]
                    if args.mixup:
                        t_b = lam * t_b + (1 - lam) * t_b[mp]
                loss = loss + args.elr_lambda * torch.log(1 - (t_b * p).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            if args.sop:
                opt_uv.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            if args.sop:
                scaler.step(opt_uv)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(bi)
            seen += len(bi)

        # ---- evaluation (eval mode, no augmentation)
        lv = predict(net, D["x_val"], D["mean"], D["std"], amp)
        lt = predict(net, D["x_test"], D["mean"], D["std"], amp)
        val_logits[ep - 1] = lv.cpu().numpy()
        test_logits[ep - 1] = lt.cpu().numpy()
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=tot_loss / seen)
        if holdout_logits is not None:
            holdout_logits[ep - 1] = predict(net, D["x_holdout"], D["mean"], D["std"], amp).cpu().numpy()
        if train_logits is not None:
            ltr = predict(net, D["x_train"], D["mean"], D["std"], amp)
            train_logits[ep - 1] = ltr.cpu().numpy()
            row |= dict(
                train_acc_given=acc(ltr, D["y_train"]),
                train_acc_correct=acc(ltr, D["y_train"], correct_mask),
                train_acc_mislabeled=acc(ltr, D["y_train"], ~correct_mask),
                # on mislabeled images: how often the net still predicts the true class (measurement only)
                train_mislabeled_pred_clean=acc(ltr, D["train_clean"], ~correct_mask),
            )
        row |= dict(
            val_acc_given=acc(lv, D["y_val"]),  # <- model selection uses this
            val_acc_noisy=acc(lv, D["val_noisy"]),
            val_acc_clean=acc(lv, D["val_clean"]),
            val_acc_synthetic=acc(lv, D["val_synth"]),
            time_s=time.time() - t0,
        )
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        pd.DataFrame(log).to_csv(os.path.join(run_dir, "log.csv"), index=False)

    # ---- outputs
    df = pd.DataFrame(log)
    best = int(df.loc[df.val_acc_given.idxmax(), "epoch"])  # first epoch with the max, selection on given labels
    write_preds(os.path.join(run_dir, "pred_final.csv"), test_logits[-1])
    write_preds(os.path.join(run_dir, "pred_best.csv"), test_logits[best - 1])
    if train_logits is not None:
        df[["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]] \
            .to_csv(os.path.join(run_dir, "dynamics.csv"), index=False)
        np.save(os.path.join(run_dir, "train_logits.npy"), train_logits)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "val_logits.npy"), val_logits)
    np.save(os.path.join(run_dir, "train_index.npy"), D["train_idx"])
    if holdout_logits is not None:
        np.save(os.path.join(run_dir, "holdout_logits.npy"), holdout_logits)
        np.save(os.path.join(run_dir, "holdout_index.npy"), D["holdout_idx"])
    torch.save(net.state_dict(), os.path.join(run_dir, "model_final.pt"))

    summary = dict(best_epoch=best, val_acc_given_best=df.val_acc_given.max(),
                   val_acc_given_final=df.val_acc_given.iloc[-1],
                   val_acc_noisy_final=df.val_acc_noisy.iloc[-1],
                   val_acc_clean_final=df.val_acc_clean.iloc[-1],
                   total_time_min=df.time_s.sum() / 60)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
