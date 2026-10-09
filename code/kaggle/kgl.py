"""Run training on Kaggle GPUs from this computer, via the Kaggle CLI.

One-time:  kaggle auth login                      (or save an API token to ~/.kaggle/access_token)
           python kaggle/kgl.py data              upload data/ as a PRIVATE dataset (re-run = new version)
           python kaggle/kgl.py aux               upload outputs/aux/ (Task 4 inputs) as a PRIVATE dataset
Per job:   python kaggle/kgl.py push task1        build the notebook from code/ and run it on GPU T4 x2
           python kaggle/kgl.py status task1      queued / running / complete / error
           python kaggle/kgl.py fetch task1       download outputs into outputs/ (-> outputs/runs/<name>/)

The Kaggle username is taken from --user, else $KAGGLE_USERNAME, else `kaggle config view`.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# project root = folder holding data/ and code/ (kaggle/ sits in it, or in code/ inside the submission zip)
_up = os.path.dirname(HERE)
CV = _up if os.path.isdir(os.path.join(_up, "data")) else os.path.dirname(_up)
sys.path.insert(0, HERE)
from make_notebook import JOBS, build  # noqa: E402

# v1 ("cv-bootcamp-data", uploaded as loose files) failed to mount on Kaggle (ERRORED_MOUNTING_DATASET);
# v2 is uploaded as a single zip, which Kaggle extracts into data/.
DATASET_SLUG = "cv-bootcamp-data2"
AUX_SLUG = "cv-bootcamp-aux"


def kaggle(*args, capture=False):
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("$ kaggle " + " ".join(args), flush=True)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    return subprocess.run(cmd).returncode, ""


def username(cli_user):
    if cli_user:
        return cli_user
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    _, out = kaggle("config", "view", capture=True)
    m = re.search(r"username:\s*(\S+)", out)
    if not m or m.group(1).lower() == "none":
        sys.exit("could not find your Kaggle username; pass --user <name>")
    return m.group(1)


def kernel_slug(jobs):
    return f"cv-bootcamp-{jobs}"


def cmd_data(user):
    folder = os.path.join(HERE, "dataset_zip")
    os.makedirs(folder, exist_ok=True)
    zpath = os.path.join(folder, "cv-bootcamp-data.zip")
    if not os.path.exists(zpath):
        from make_notebook import zip_data
        zip_data(zpath)
    meta = {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "unknown"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    code, out = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", capture=True)
    if code == 0 and "ready" in out.lower():
        rc, _ = kaggle("datasets", "version", "-p", folder, "-m", "update", "-t", "-r", "skip")
    else:  # private by default (no --public flag)
        rc, _ = kaggle("datasets", "create", "-p", folder, "-t", "-r", "skip")
    print(f"\nDataset: https://www.kaggle.com/datasets/{user}/{DATASET_SLUG}  (private)")
    print("Kaggle needs a minute or two to process it; check with: "
          f"kaggle datasets status {user}/{DATASET_SLUG}")
    return rc


def cmd_aux(user):
    """Upload outputs/aux/ (Task 4 inputs: targets .npy, index subsets .txt) as a second private dataset."""
    folder = os.path.join(CV, "outputs", "aux")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "AUX_MARKER.txt"), "w") as fh:
        fh.write("marker so the notebook can find this dataset\n")
    meta = {"title": AUX_SLUG, "id": f"{user}/{AUX_SLUG}", "licenses": [{"name": "unknown"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    code, out = kaggle("datasets", "status", f"{user}/{AUX_SLUG}", capture=True)
    if code == 0 and "ready" in out.lower():
        rc, _ = kaggle("datasets", "version", "-p", folder, "-m", "update", "-t", "-r", "skip")
    else:
        rc, _ = kaggle("datasets", "create", "-p", folder, "-t", "-r", "skip")
    return rc


def cmd_push(user, jobs):
    folder = os.path.join(HERE, f"kernel_{jobs}")
    os.makedirs(folder, exist_ok=True)
    nb = f"cv_{jobs}.ipynb"
    build(jobs, os.path.join(folder, nb))
    meta = {
        "id": f"{user}/{kernel_slug(jobs)}", "title": kernel_slug(jobs), "code_file": nb,
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",  # = GPU T4 x2
        "dataset_sources": [f"{user}/{DATASET_SLUG}"]
        + ([f"{user}/{AUX_SLUG}"] if any("{AUX}" in j[1] or "{FILE:" in j[1] for j in JOBS[jobs]) else []),
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    rc, _ = kaggle("kernels", "push", "-p", folder)
    print(f"\nNotebook: https://www.kaggle.com/code/{user}/{kernel_slug(jobs)}")
    return rc


def cmd_status(user, jobs):
    return kaggle("kernels", "status", f"{user}/{kernel_slug(jobs)}")[0]


def cmd_fetch(user, jobs):
    dest = os.path.join(CV, "outputs")
    os.makedirs(dest, exist_ok=True)
    rc, _ = kaggle("kernels", "output", f"{user}/{kernel_slug(jobs)}", "-p", dest, "-o")
    print(f"\nDownloaded into {dest} (runs are in outputs/runs/<name>/)")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["data", "aux", "push", "status", "fetch"])
    ap.add_argument("jobs", nargs="?", default="task1", choices=list(JOBS))
    ap.add_argument("--user", default=None)
    a = ap.parse_args()
    user = username(a.user)
    rc = {"data": lambda: cmd_data(user), "aux": lambda: cmd_aux(user),
          "push": lambda: cmd_push(user, a.jobs),
          "status": lambda: cmd_status(user, a.jobs),
          "fetch": lambda: cmd_fetch(user, a.jobs)}[a.action]()
    sys.exit(rc)


if __name__ == "__main__":
    main()
