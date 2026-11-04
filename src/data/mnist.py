"""Download (with checksum), parse and split MNIST."""
import gzip, hashlib, urllib.request
from pathlib import Path
import numpy as np

FILES = {
    "train-images-idx3-ubyte.gz": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", 9912422),
    "train-labels-idx1-ubyte.gz": ("d53e105ee54ea40749a09fcbcd1e9432", 28881),
    "t10k-images-idx3-ubyte.gz": ("9fb629c4189551a2d022fa330f9573f3", 1648877),
    "t10k-labels-idx1-ubyte.gz": ("ec29112dd5afa0611ce80d1b7f02629c", 4542),
}
# The original host is tried last: it is frequently unreachable. Every file is
# verified against the official MD5, so any mirror that works is equally valid.
MIRRORS = [
    "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    used = {}
    for name, (md5, _size) in FILES.items():
        dest = raw_dir / name
        if dest.exists() and _md5(dest) == md5:
            used[name] = "local copy"
            continue
        for base in MIRRORS:
            try:
                urllib.request.urlretrieve(base + name, dest)
            except Exception as e:                      # try next mirror
                print(f"  {base}{name}: {type(e).__name__}")
                continue
            if _md5(dest) == md5:
                used[name] = base
                break
            print(f"  {base}{name}: checksum mismatch")
            dest.unlink()
        else:
            raise RuntimeError(f"Could not download {name}. Put the four official gz files "
                               f"in {raw_dir} manually.")
    return used


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[:4], "big") == 2051
    n, r, c = (int.from_bytes(data[i:i + 4], "big") for i in (4, 8, 12))
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, r, c)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[:4], "big") == 2049
    return np.frombuffer(data, np.uint8, offset=8)


def load_splits():
    from config import DATA_DIR
    z = np.load(DATA_DIR / "mnist_1to9.npz")
    return {k: z[k] for k in z.files}
