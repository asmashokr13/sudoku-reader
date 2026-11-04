"""9x9 image mosaics of handwritten digits and injection of hard (misread) digits.

Pools (built from the MNIST *test* split only, using the trained network):
  clean[d] : test images of digit d that the network reads correctly
  hard[d]  : test images of digit d that the network misreads but where d is
             still among its top-3 digits  (a recoverable misread)
Normal given cells use a random clean image, so the number of misreads in a puzzle is
exactly the number injected. A misread cell uses a random hard image of the true digit.
"""
from dataclasses import dataclass
import numpy as np
from config import BLANK_THRESHOLD

BLANK_NOISE_SD = 0.01


def blank_image(rng: np.random.Generator):
    return np.clip(rng.normal(0, BLANK_NOISE_SD, (28, 28)), 0, 1).astype(np.float32)


def is_blank(img) -> bool:
    """Brightness rule (not the NN): mean pixel value below BLANK_THRESHOLD."""
    return float(np.mean(img)) < BLANK_THRESHOLD


class ImagePools:
    def __init__(self, labels, probs):
        """labels: digits 1..9 of the test images; probs: (N,9) network probabilities."""
        pred = probs.argmax(1) + 1
        top3 = np.argsort(-probs, axis=1)[:, :3] + 1
        in_top3 = (top3 == labels[:, None]).any(1)
        self.clean = {d: np.where((labels == d) & (pred == d))[0] for d in range(1, 10)}
        self.hard = {d: np.where((labels == d) & (pred != d) & in_top3)[0] for d in range(1, 10)}


@dataclass
class Instance:
    puzzle: list            # 81 digits, 0 = blank (ground truth givens)
    solution: list          # 81 digits (ground truth)
    img_idx: np.ndarray     # 81 indices into the test set, -1 for blank cells
    injected: list          # cells whose image is a hard (misread) digit
    seed: int = 0

    def images(self, x_test, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        return np.stack([x_test[i] if i >= 0 else blank_image(rng) for i in self.img_idx])


def build_instance(puzzle, solution, pools: ImagePools, n_misreads: int, rng: np.random.Generator,
                   seed: int = 0) -> Instance:
    givens = [i for i, v in enumerate(puzzle) if v]
    ok = [i for i in givens if len(pools.hard[puzzle[i]])]
    if n_misreads > len(ok):
        raise ValueError("not enough cells with a hard image available")
    injected = sorted(rng.choice(ok, n_misreads, replace=False).tolist()) if n_misreads else []
    idx = np.full(81, -1)
    for i in givens:
        pool = pools.hard[puzzle[i]] if i in injected else pools.clean[puzzle[i]]
        idx[i] = rng.choice(pool)
    return Instance(list(puzzle), list(solution), idx, injected, seed)
