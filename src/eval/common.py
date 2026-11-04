"""Shared helpers for the experiments and the demo."""
import random
import numpy as np
from config import ARTIFACT_DIR, TEAM_SEED
from src.data.mnist import load_splits
from src.data.mosaic import ImagePools, build_instance
from src.data.sudoku_gen import make_puzzle
from src.nn.mlp import MLP


def load_world():
    """Returns (model, x_test float images, y_test digits 1-9, calibrated test probs, pools)."""
    d = load_splits()
    model = MLP.load(ARTIFACT_DIR / "mlp.npz")
    x = d["x_test"].astype(np.float32) / 255
    y = d["y_test"].astype(int)
    probs = model.predict_proba(x)
    return model, x, y, probs, ImagePools(y, probs)


def seed_for(experiment: int, seed_idx: int) -> int:
    return TEAM_SEED * 100_000 + experiment * 1_000 + seed_idx


def make_instances(pools, seed, n_puzzles, n_givens, n_misreads_list):
    """For one seed: n_puzzles puzzles; each is evaluated at every misread level
    (same puzzle, different injected cells) -> paired design."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = {k: [] for k in n_misreads_list}
    for _ in range(n_puzzles):
        p, s = make_puzzle(rng, n_givens)
        for k in n_misreads_list:
            out[k].append(build_instance(p, s, pools, k, nrng, seed))
    return out


def recovered(instance, result) -> dict:
    """Exact-solution recovery and exact-correction recovery for one pipeline result."""
    solved = result.status == "solved"
    exact_sol = solved and result.solution == instance.solution
    fixed = {c["cell"]: c["new"] for c in result.corrections}
    exact_corr = solved and set(fixed) == set(instance.injected) and \
        all(fixed[c] == instance.solution[c] for c in fixed)
    return dict(solved=solved, exact_solution=exact_sol, exact_corrections=exact_corr)


def mean_std(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
