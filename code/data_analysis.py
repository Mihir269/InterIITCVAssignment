"""Final data analysis: labels, images, training dynamics, model selection, the Task 3 detector and the final
model. Measurement only – nothing is trained or selected here; clean labels are used to *measure*.

    python code/data_analysis.py --out results/analysis

Writes figures (PNG) and analysis.json with every number quoted in docs/ANALYSIS.md.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.path.join(CV, "outputs", "runs")

# reference palette (categorical slots in fixed order), text and surface tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                     "lines.linewidth": 2})


def softmax(z):
    z = np.asarray(z, np.float32)
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=140)
    plt.close(fig)


def bars(ax, labels, series, names, colors, ylabel):
    k = len(series)
    w = 0.8 / k
    x = np.arange(len(labels))
    for i, (s, n, c) in enumerate(zip(series, names, colors)):
        ax.bar(x + (i - (k - 1) / 2) * w, s, w * 0.92, color=c, label=n, edgecolor=SURF, linewidth=1)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=k)


def end_labels(ax, x, items, gap):
    """Direct labels at line ends, nudged apart so they never overlap. items = [(y, text)]."""
    ys = sorted(items)
    placed = []
    for y, t in ys:
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.text(x, y, t, color=INK2, va="center", fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CV, "results", "analysis"))
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    R = {}

    d = os.path.join(CV, "data")
    L = pd.read_csv(os.path.join(d, "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(d, "train_idx.txt"), dtype=np.int64)
    va = np.loadtxt(os.path.join(d, "val_idx.txt"), dtype=np.int64)
    names = open(os.path.join(d, "classes.txt")).read().split()
    short = [n[:5] for n in names]
    clean, noisy, synth = (L[c].values for c in ["clean_label", "noisy_label", "synthetic_label"])
    ct, nt, st = clean[tr], noisy[tr], synth[tr]
    wrong_h, wrong_s = nt != ct, st != ct

    # ------------------------------------------------------------ A. labels
    cnt = {k: np.bincount(v, minlength=10) for k, v in [("clean", ct), ("human", nt), ("synthetic", st)]}
    R["A_class_counts"] = {k: dict(zip(names, map(int, v))) for k, v in cnt.items()}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars(ax, names, [cnt["clean"], cnt["human"], cnt["synthetic"]], ["true labels", "human labels", "synthetic labels"],
         C[:3], "training images")
    ax.set_title("A1. How many training images carry each label")
    save(fig, out, "A1_class_counts.png")

    rate_h = np.array([wrong_h[ct == k].mean() for k in range(10)])
    rate_s = np.array([wrong_s[ct == k].mean() for k in range(10)])
    order = np.argsort(-rate_h)
    R["A_noise_rate_by_true_class"] = {names[k]: dict(human=round(float(rate_h[k]), 4), synthetic=round(float(rate_s[k]), 4))
                                       for k in order}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars(ax, [names[k] for k in order], [rate_h[order] * 100, rate_s[order] * 100], ["human", "synthetic"], C[:2],
         "% of the class labelled wrongly")
    ax.set_title("A2. Noise rate per true class")
    save(fig, out, "A2_noise_rate_per_class.png")

    M = np.zeros((10, 10), int)
    np.add.at(M, (ct, nt), 1)
    pairs = sorted([(M[i, j] + M[j, i], i, j) for i in range(10) for j in range(i + 1, 10)], reverse=True)[:6]
    R["A_pair_asymmetry"] = [dict(pair=f"{names[i]}<->{names[j]}", i_to_j=int(M[i, j]), j_to_i=int(M[j, i]))
                             for _, i, j in pairs]
    # purity of each given label: share of images labelled k that truly are k
    pur_h = np.array([(ct[nt == k] == k).mean() for k in range(10)])
    pur_s = np.array([(ct[st == k] == k).mean() for k in range(10)])
    R["A_label_purity"] = {names[k]: dict(human=round(float(pur_h[k]), 4), synthetic=round(float(pur_s[k]), 4))
                           for k in range(10)}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    o = np.argsort(pur_h)
    bars(ax, [names[k] for k in o], [pur_h[o] * 100, pur_s[o] * 100], ["human", "synthetic"], C[:2],
         "% of images with this label that truly are this class")
    ax.set_ylim(0, 100)
    ax.set_title("A3. How trustworthy is each given label?")
    save(fig, out, "A3_label_purity.png")

    # ------------------------------------------------------------ B. images
    z = np.load(os.path.join(RUNS, "t1_clean", "val_logits.npy"))[-1]
    p = softmax(z)
    cv = clean[va]
    conf = p[np.arange(len(va)), cv]                      # difficulty = clean model's confidence in the true class
    q = pd.qcut(conf, 10, labels=False, duplicates="drop")
    wv_h, wv_s = noisy[va] != cv, synth[va] != cv
    dec_h = [wv_h[q == b].mean() for b in range(q.max() + 1)]
    dec_s = [wv_s[q == b].mean() for b in range(q.max() + 1)]
    R["B_mislabel_rate_by_difficulty_decile"] = dict(human=[round(float(x), 4) for x in dec_h],
                                                     synthetic=[round(float(x), 4) for x in dec_s])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(1, len(dec_h) + 1)
    ax.plot(xs, np.array(dec_h) * 100, color=C[0], marker="o", ms=5, label="human")
    ax.plot(xs, np.array(dec_s) * 100, color=C[1], marker="o", ms=5, label="synthetic")
    ax.set_xticks(xs, ["hardest"] + [str(i) for i in xs[1:-1]] + ["easiest"])
    ax.set_xlabel("validation images in difficulty deciles (clean-label model's confidence in the true class)")
    ax.set_ylabel("% mislabelled")
    ax.legend(loc="upper right")
    ax.set_title("B1. Humans mislabel the hard images; random noise does not care")
    save(fig, out, "B1_mislabel_vs_difficulty.png")

    imgs = np.load(os.path.join(d, "train_images.npy"), mmap_mode="r")
    X = np.asarray(imgs[tr]).astype(np.float32) / 255
    gray = X.mean(-1)
    bright = gray.mean((1, 2))
    contrast = gray.std((1, 2))
    rg, yb = X[..., 0] - X[..., 1], 0.5 * (X[..., 0] + X[..., 1]) - X[..., 2]
    colorful = np.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2) + 0.3 * np.sqrt(rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2)
    stats = {}
    for nm, v in [("brightness", bright), ("contrast", contrast), ("colourfulness", colorful)]:
        stats[nm] = dict(human_wrong=float(v[wrong_h].mean()), human_correct=float(v[~wrong_h].mean()),
                         synth_wrong=float(v[wrong_s].mean()), synth_correct=float(v[~wrong_s].mean()),
                         auroc_human=float(roc_auc_score(wrong_h, -v)), auroc_synth=float(roc_auc_score(wrong_s, -v)))
    R["B_image_statistics"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in stats.items()}
    # mislabel rate by contrast decile (low contrast first)
    qc = pd.qcut(contrast, 10, labels=False)
    R["B_mislabel_rate_by_contrast_decile"] = dict(human=[round(float(wrong_h[qc == b].mean()), 4) for b in range(10)],
                                                   synthetic=[round(float(wrong_s[qc == b].mean()), 4) for b in range(10)])
    del X, gray

    # ------------------------------------------------------------ C. training dynamics
    ln, ls = pd.read_csv(os.path.join(RUNS, "t1_noisy", "log.csv")), pd.read_csv(os.path.join(RUNS, "t5_baseline_synth", "log.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(ln.epoch, ln.train_acc_mislabeled * 100, color=C[0], label="human labels")
    ax.plot(ls.epoch, ls.train_acc_mislabeled * 100, color=C[1], label="synthetic labels")
    end_labels(ax, 61, [(float(ln.train_acc_mislabeled.iloc[-1] * 100), "human"),
                        (float(ls.train_acc_mislabeled.iloc[-1] * 100), "synthetic")], 4)
    ax.set_xlim(0, 68)
    ax.set_xlabel("epoch")
    ax.set_ylabel("% of mislabelled training images fitted")
    ax.legend(loc="upper left")
    ax.set_title("C1. Plain CE memorizes human wrong labels sooner and more")
    save(fig, out, "C1_memorization_real_vs_synthetic.png")
    R["C_memorization_final"] = dict(human=round(float(ln.train_acc_mislabeled.iloc[-1]), 4),
                                     synthetic=round(float(ls.train_acc_mislabeled.iloc[-1]), 4))
    for nm, lg in [("human", ln), ("synthetic", ls)]:
        m = lg.train_acc_mislabeled.values
        R[f"C_epoch_mislabeled_fit_exceeds_30pct_{nm}"] = int(lg.epoch[np.argmax(m > 0.30)]) if (m > 0.30).any() else None

    TL = np.load(os.path.join(RUNS, "t1_noisy", "train_logits.npy"), mmap_mode="r")
    pred60 = np.asarray(TL[-1]).argmax(1)
    mem_cls = np.array([(pred60[wrong_h & (ct == k)] == nt[wrong_h & (ct == k)]).mean() for k in range(10)])
    R["C_memorized_share_by_true_class_epoch60"] = {names[k]: round(float(mem_cls[k]), 4) for k in range(10)}

    curves = {"CE": "t1_noisy", "ELR + mixup": "t4_elr_mix1_ens", "relabel + mixup": "t4_cvrelabel_mix2_ens",
              "GCE + mixup": "t4_gce_mixup_ens"}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ends = []
    for (nm, r), c in zip(curves.items(), C):
        lg = pd.read_csv(os.path.join(RUNS, r, "log.csv"))
        ax.plot(lg.epoch, lg.val_acc_clean * 100, color=c, label=nm)
        ends.append((float(lg.val_acc_clean.iloc[-1] * 100), nm))
    end_labels(ax, 61, ends, 2.6)
    ax.set_xlim(0, 75)
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean validation accuracy (%)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("C2. Robust methods keep improving where CE collapses (measurement only)")
    save(fig, out, "C2_clean_val_curves.png")

    # ------------------------------------------------------------ D. model selection
    runs = pd.read_csv(os.path.join(CV, "results", "tables", "runs_noisy.csv")) \
        if os.path.exists(os.path.join(CV, "results", "tables", "runs_noisy.csv")) else None
    if runs is None:
        rows = []
        for dd in sorted(os.listdir(RUNS)):
            f = os.path.join(RUNS, dd, "log.csv")
            cfgf = os.path.join(RUNS, dd, "config.json")
            if os.path.isfile(f) and os.path.isfile(cfgf):
                cfg = json.load(open(cfgf))
                if cfg.get("label") != "noisy" or cfg.get("cv_folds"):
                    continue
                lg = pd.read_csv(f)
                b = lg.loc[lg.val_acc_given.idxmax()]
                rows.append(dict(run=dd, val_acc_given_best=b.val_acc_given, val_acc_clean_at_best=b.val_acc_clean))
        runs = pd.DataFrame(rows)
    runs = runs[runs.cv_fold.isna()] if "cv_fold" in runs else runs
    sub = {"t4_elr_mix1_ens", "t4_cvrelabel_mix2_ens", "t4_gce_mixup_ens"}
    isub = runs.run.isin(sub)
    idm = runs.run == "t4_dividemix"
    oth = ~(isub | idm)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(runs.val_acc_given_best[oth] * 100, runs.val_acc_clean_at_best[oth] * 100, s=34, color=C[0],
               edgecolor=SURF, linewidth=1.5, label="other runs")
    ax.scatter(runs.val_acc_given_best[isub] * 100, runs.val_acc_clean_at_best[isub] * 100, s=60, color=C[1],
               edgecolor=SURF, linewidth=1.5, label="submitted (Task 4)")
    ax.scatter(runs.val_acc_given_best[idm] * 100, runs.val_acc_clean_at_best[idm] * 100, s=60, color=C[2],
               edgecolor=SURF, linewidth=1.5, label="DivideMix")
    for _, r in runs[isub | idm | (runs.run == "t1_noisy")].iterrows():
        lab = {"t4_elr_mix1_ens": "ELR + mixup", "t4_cvrelabel_mix2_ens": "relabel + mixup",
               "t4_gce_mixup_ens": "GCE + mixup", "t4_dividemix": "DivideMix", "t1_noisy": "CE (early stop)"}[r.run]
        ax.annotate(lab, (r.val_acc_given_best * 100, r.val_acc_clean_at_best * 100), textcoords="offset points",
                    xytext=(6, -3), fontsize=8.5, color=INK2)
    ax.set_xlabel("noisy validation accuracy (%) – the selection signal")
    ax.set_ylabel("clean validation accuracy (%) – measurement")
    ax.legend(loc="lower right")
    ax.set_title("D1. Noisy validation tracks true accuracy, but compresses the top")
    save(fig, out, "D1_noisy_vs_clean_val.png")
    from scipy.stats import spearmanr
    R["D_spearman_noisy_vs_clean"] = round(float(spearmanr(runs.val_acc_given_best, runs.val_acc_clean_at_best)[0]), 4)
    top = runs[runs.val_acc_given_best >= 0.58]
    R["D_top_runs_noisy_val_range"] = [round(float(top.val_acc_given_best.min()), 4), round(float(top.val_acc_given_best.max()), 4)]
    R["D_top_runs_clean_val_range"] = [round(float(top.val_acc_clean_at_best.min()), 4), round(float(top.val_acc_clean_at_best.max()), 4)]
    R["D_n_runs"] = int(len(runs))

    # ------------------------------------------------------------ E. Task 3 detector
    sc = pd.read_csv(os.path.join(CV, "predictions", "t3_noise_scores.csv")).set_index("index").loc[tr].score.values
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(sc.min(), sc.max(), 60)
    ax.hist(sc[~wrong_h], bins, color=C[0], alpha=0.75, label="correct label", edgecolor=SURF, linewidth=0.5)
    ax.hist(sc[wrong_h], bins, color=C[1], alpha=0.75, label="wrong label", edgecolor=SURF, linewidth=0.5)
    ax.set_xlabel("AUM noise score (higher = label more likely wrong)")
    ax.set_ylabel("training images")
    ax.legend(loc="upper center")
    ax.set_title("E1. The submitted noise score separates wrong from correct labels")
    save(fig, out, "E1_noise_score_distribution.png")
    R["E_auroc_overall"] = round(float(roc_auc_score(wrong_h, sc)), 4)
    R["E_auroc_by_given_label"] = {names[k]: round(float(roc_auc_score(wrong_h[nt == k], sc[nt == k])), 4) for k in range(10)}
    order = np.argsort(-sc)
    budgets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    R["E_precision_recall_at_budget"] = [dict(review_share=b, precision=round(float(wrong_h[order[:int(b * len(sc))]].mean()), 4),
                                              recall=round(float(wrong_h[order[:int(b * len(sc))]].sum() / wrong_h.sum()), 4))
                                         for b in budgets]
    pr = R["E_precision_recall_at_budget"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot([b["review_share"] * 100 for b in pr], [b["precision"] * 100 for b in pr], color=C[0], marker="o", ms=5,
            label="precision (share of flagged that are wrong)")
    ax.plot([b["review_share"] * 100 for b in pr], [b["recall"] * 100 for b in pr], color=C[1], marker="o", ms=5,
            label="recall (share of all wrong labels found)")
    ax.axvline(40.24, color=INK2, lw=1, ls="--")
    ax.text(40.8, 60, "true noise rate (40%)", color=INK2, fontsize=8.5)
    ax.set_xlabel("% of training images flagged (highest scores first)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.82))
    ax.set_title("E2. Reviewing the top 20% of scores catches about half of all wrong labels")
    save(fig, out, "E2_precision_recall_budget.png")

    # ------------------------------------------------------------ F. final model
    def val_probs(run):
        lg = pd.read_csv(os.path.join(RUNS, run, "log.csv"))
        e = int(lg.loc[lg.val_acc_given.idxmax(), "epoch"])
        return softmax(np.load(os.path.join(RUNS, run, "val_logits.npy"))[e - 1])
    pce, pel = val_probs("t1_noisy").argmax(1), val_probs("t4_elr_mix1_ens").argmax(1)
    acc_ce = np.array([(pce[cv == k] == k).mean() for k in range(10)])
    acc_el = np.array([(pel[cv == k] == k).mean() for k in range(10)])
    R["F_per_class_clean_val_acc"] = {names[k]: dict(ce_early_stop=round(float(acc_ce[k]), 4),
                                                     elr_mixup=round(float(acc_el[k]), 4)) for k in range(10)}
    o = np.argsort(acc_el)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars(ax, [names[k] for k in o], [acc_ce[o] * 100, acc_el[o] * 100], ["CE, early stop", "ELR + mixup (best)"], C[:2],
         "clean validation accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("F1. Per-class accuracy: big gains everywhere except cat (cat→dog bias)")
    save(fig, out, "F1_per_class_accuracy.png")

    CM = np.zeros((10, 10), int)
    np.add.at(CM, (cv, pel), 1)
    off = [(CM[i, j], names[i], names[j]) for i in range(10) for j in range(10) if i != j]
    R["F_top_remaining_confusions_val"] = [dict(true=t, predicted=pp, count=int(c)) for c, t, pp in sorted(off, reverse=True)[:6]]
    fig, ax = plt.subplots(figsize=(5.8, 5))
    nrm = CM / CM.sum(1, keepdims=True)
    ax.imshow(nrm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{nrm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if nrm[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), short, rotation=45)
    ax.set_yticks(range(10), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title("F2. Best model on validation (% of each true class)")
    save(fig, out, "F2_best_model_confusion.png")

    TE = np.load(os.path.join(RUNS, "s_elr_mix1", "train_logits.npy"), mmap_mode="r")
    pe = np.asarray(TE[57]).argmax(1)  # epoch 58 = the selected epoch
    R["F_mislabeled_train_images_corrected_by_class"] = {
        names[k]: round(float((pe[wrong_h & (ct == k)] == k).mean()), 4) for k in range(10)}
    R["F_mislabeled_train_images_corrected_overall"] = round(float((pe[wrong_h] == ct[wrong_h]).mean()), 4)
    R["F_correct_train_images_kept"] = round(float((pe[~wrong_h] == ct[~wrong_h]).mean()), 4)
    R["F_ce_epoch60_mislabeled_true_class"] = round(float((pred60[wrong_h] == ct[wrong_h]).mean()), 4)

    files = ["t1_clean_ref", "t1_noisy_ce", "t2_early_stop", "t4_elr_mixup", "t4_cv_relabel_mixup", "t4_gce_mixup",
             "t5_baseline_synth", "t5_best_synth"]
    P = {f: pd.read_csv(os.path.join(CV, "predictions", f + ".csv")).sort_values("id").pred.values for f in files}
    R["F_test_predicted_class_counts"] = {f: dict(zip(names, map(int, np.bincount(P[f], minlength=10)))) for f in files}
    R["F_test_agreement_with_clean_reference"] = {f: round(float((P[f] == P["t1_clean_ref"]).mean()), 4) for f in files}
    R["F_test_agreement_matrix"] = {f: {g: round(float((P[f] == P[g]).mean()), 4) for g in files} for f in files}

    with open(os.path.join(out, "analysis.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R, indent=1)[:6000])


if __name__ == "__main__":
    main()
