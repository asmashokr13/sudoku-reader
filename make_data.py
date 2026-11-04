#!/usr/bin/env python
"""Reproducible data preparation.

1. download the 4 official MNIST gz files (MD5-verified, several mirrors)
2. split the 60,000 training images 50,000 train / 10,000 validation (fixed seed)
3. drop digit 0 in every split (the Sudoku uses 1-9)
4. save uint8 arrays to data/mnist_1to9.npz  (pixels are scaled to 0-1 at load time)
5. write data/manifest.json (counts, checksums, brightness statistics)
"""
import json, platform, sys
import numpy as np
from config import DATA_DIR, TEAM_SEED, BLANK_THRESHOLD
from src.data import mnist

SPLIT_SEED = TEAM_SEED * 1000 + 1


def main():
    raw = DATA_DIR / "raw"
    print("Downloading / verifying MNIST ...")
    used = mnist.download_all(raw)
    xtr = mnist.read_idx_images(raw / "train-images-idx3-ubyte.gz")
    ytr = mnist.read_idx_labels(raw / "train-labels-idx1-ubyte.gz")
    xte = mnist.read_idx_images(raw / "t10k-images-idx3-ubyte.gz")
    yte = mnist.read_idx_labels(raw / "t10k-labels-idx1-ubyte.gz")
    assert xtr.shape == (60000, 28, 28) and xte.shape == (10000, 28, 28)

    perm = np.random.default_rng(SPLIT_SEED).permutation(60000)
    tr_idx, va_idx = perm[:50000], perm[50000:]
    out = {}
    for name, x, y, idx in [("train", xtr, ytr, tr_idx), ("val", xtr, ytr, va_idx),
                            ("test", xte, yte, np.arange(10000))]:
        xs, ys = x[idx], y[idx]
        keep = ys != 0
        out[f"x_{name}"], out[f"y_{name}"] = xs[keep], ys[keep]
        print(f"  {name}: {len(idx)} images -> {keep.sum()} after dropping digit 0")
    np.savez_compressed(DATA_DIR / "mnist_1to9.npz", **out)

    means = {k: float((out[f"x_{k}"].mean(axis=(1, 2)) / 255).min()) for k in ("train", "val", "test")}
    manifest = {
        "split_seed": SPLIT_SEED, "mirrors_used": used,
        "md5": {n: m for n, (m, _) in mnist.FILES.items()},
        "counts": {k: int(len(out[f"y_{k}"])) for k in ("train", "val", "test")},
        "min_digit_mean_brightness": means, "blank_threshold": BLANK_THRESHOLD,
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"]), "min digit brightness:", means)
    assert min(means.values()) > BLANK_THRESHOLD, "blank rule would confuse a thin digit with a blank"


if __name__ == "__main__":
    main()
