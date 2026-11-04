import random
import numpy as np
from src.data.sudoku_gen import make_puzzle
from src.csp.sudoku_csp import PEERS, find_conflicts
from src.integration.pipeline import Reading

CLASSIC = ("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
CLASSIC_SOLUTION = ("534678912672195348198342567859761423426853791713924856961537284287419635345286179")


def digits(s):
    return [int(ch) for ch in s]


def synthetic_reading(puzzle, wrong=None, seed=0, wrong_conf=(0.52, 0.45), sure=0.97):
    """Reading with known probabilities. wrong = {cell: wrongly read digit}.
    Correct cells: true digit with prob `sure`. Wrong cells: wrong digit 0.52, true digit 0.45
    (the example of the project statement)."""
    wrong = wrong or {}
    rng = np.random.default_rng(seed)
    probs = np.zeros((81, 9), np.float32)
    grid = [0] * 81
    for c, v in enumerate(puzzle):
        if not v:
            continue
        p = rng.random(9) + 0.01
        if c in wrong:
            r, t = wrong[c], v
            p = p / p.sum() * (1 - sum(wrong_conf)); p[r - 1] = wrong_conf[0]; p[t - 1] = wrong_conf[1]
        else:
            p = p / p.sum() * (1 - sure); p[v - 1] = sure
        probs[c] = p / p.sum()
        grid[c] = int(probs[c].argmax()) + 1
    return Reading(grid, probs, [v == 0 for v in puzzle])


def conflict_wrong_digit(puzzle, cell, rng):
    """A wrong digit for `cell` that copies a given peer's digit => guaranteed duplicate."""
    opts = sorted({puzzle[p] for p in PEERS[cell] if puzzle[p] and puzzle[p] != puzzle[cell]})
    return rng.choice(opts)


def make_case(seed, givens=30):
    rng = random.Random(seed)
    return make_puzzle(rng, givens)
