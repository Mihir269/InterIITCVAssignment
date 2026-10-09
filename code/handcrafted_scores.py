"""Task 3 (extra): noise scores from HAND-CRAFTED features (no training at all), scored with the same
kNN / centroid outlier signals as code/feature_scores.py.

Features per image (32x32 RGB):
  color  HSV histograms (16 hue, 8 saturation, 8 value bins)                      32 dims
  lbp    uniform LBP (P=8, R=1) histogram on grayscale                              10 dims
  glcm   gray-level co-occurrence (32 levels, distance 1, 4 angles):
         contrast, homogeneity, energy, correlation, dissimilarity, ASM            24 dims
  hog    HOG (9 orientations, 8x8 cells, 2x2 blocks)                               324 dims
Each block is z-scored, the concatenation is reduced with PCA (whitened) and L2-normalised.

    python code/handcrafted_scores.py
Clean labels are used only for the AUROC / precision@k check.
"""
import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.color import rgb2gray, rgb2hsv  # noqa: E402
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from feature_scores import scores_for  # noqa: E402


def color_hist(img):
    hsv = rgb2hsv(img)
    return np.concatenate([np.histogram(hsv[..., 0], 16, (0, 1))[0], np.histogram(hsv[..., 1], 8, (0, 1))[0],
                           np.histogram(hsv[..., 2], 8, (0, 1))[0]]) / 1024.0


def lbp_hist(gray_u8):
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    return np.histogram(lbp, bins=10, range=(0, 10))[0] / lbp.size


def glcm_feats(gray_u8):
    q = (gray_u8 // 8).astype(np.uint8)  # 32 grey levels
    m = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                     symmetric=True, normed=True)
    return np.concatenate([graycoprops(m, p).ravel() for p in
                           ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]])


def extract(imgs):
    blocks = {"color": [], "lbp": [], "glcm": [], "hog": []}
    t0 = time.time()
    for i, img in enumerate(imgs):
        g = (rgb2gray(img) * 255).astype(np.uint8)
        blocks["color"].append(color_hist(img))
        blocks["lbp"].append(lbp_hist(g))
        blocks["glcm"].append(glcm_feats(g))
        blocks["hog"].append(hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
        if i % 10000 == 0:
            print(f"  {i}/{len(imgs)}  {time.time() - t0:.0f}s", flush=True)
    return {k: np.nan_to_num(np.asarray(v, np.float32)) for k, v in blocks.items()}


def prep(x, dims):
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    if dims and x.shape[1] > dims:
        x = PCA(dims, whiten=True, random_state=0).fit_transform(x)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def main():
    d = os.path.join(CV, "data")
    labels = pd.read_csv(os.path.join(d, "labels.csv")).set_index("index").sort_index()
    tr = np.loadtxt(os.path.join(d, "train_idx.txt"), dtype=np.int64)
    imgs = np.load(os.path.join(d, "train_images.npy"), mmap_mode="r")
    cache = os.path.join(CV, "outputs", "task3", "handcrafted_features.npz")
    if os.path.exists(cache):
        B = dict(np.load(cache))
    else:
        print("extracting hand-crafted features ...")
        B = extract(np.asarray(imgs[tr]))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **B)

    y = labels.noisy_label.values[tr]
    wrong = (labels.noisy_label.values != labels.clean_label.values)[tr]  # measurement only
    k = int(wrong.sum())
    aum = np.load(os.path.join(CV, "outputs", "task3", "scores.npy"))[tr]
    sets = {"color": ["color"], "lbp": ["lbp"], "glcm": ["glcm"], "hog": ["hog"],
            "all four": ["color", "lbp", "glcm", "hog"]}
    rows = []
    for name, parts in sets.items():
        feat = prep(np.concatenate([(B[p] - B[p].mean(0)) / (B[p].std(0) + 1e-6) for p in parts], 1), 64)
        for sname, s in scores_for(feat, y, 5, 50).items():
            r = (pd.Series(s).rank(pct=True).values + pd.Series(aum).rank(pct=True).values) / 2
            rows.append(dict(features=name, signal=sname, auroc=roc_auc_score(wrong, s),
                             p_at_k=wrong[np.argsort(-s, kind="stable")[:k]].mean(),
                             auroc_with_aum=roc_auc_score(wrong, r),
                             p_at_k_with_aum=wrong[np.argsort(-r, kind="stable")[:k]].mean()))
    # how well do these features separate the classes at all? (kNN accuracy vs the NOISY labels, no clean labels)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nreference: AUM alone AUROC {roc_auc_score(wrong, aum):.4f}, "
          f"P@k {wrong[np.argsort(-aum, kind='stable')[:k]].mean():.4f}")


if __name__ == "__main__":
    main()
