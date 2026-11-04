"""Impossible cases (4): all must produce a clear 'no solution' message."""
import random
from src.csp.sudoku_csp import solve, CONFIGS, find_conflicts
from src.integration.pipeline import run_pipeline
from tests.helpers import *


def test_impossible_1_two_equal_givens_in_one_row():
    g = digits(CLASSIC); g[2] = 5                       # second 5 in row 1
    for name, cfg in CONFIGS.items():
        r = solve(g, **cfg)
        assert r.status == "no_solution" and "NO SOLUTION" in r.message, name
    assert "AC-3 emptied the domain" in solve(g, use_fc=True, use_ac3=True).message


def test_impossible_2_conflict_free_grid_with_a_cell_that_has_no_digit():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]; g[17] = 9       # cell (row1,col9) sees 1..8 in its row and 9 in its column
    assert not find_conflicts(g)
    for name, cfg in CONFIGS.items():
        r = solve(g, **cfg)
        assert r.status == "no_solution" and "NO SOLUTION" in r.message, name


def _four_disjoint_conflicts(seed):
    puz, sol = make_case(seed, 32)
    rng = random.Random(seed)
    givens = [i for i, v in enumerate(puz) if v]
    for _ in range(500):
        cells = rng.sample(givens, 4)
        wrong = {c: conflict_wrong_digit(puz, c, rng) for c in cells}
        g = puz[:]
        for c, d in wrong.items():
            g[c] = d
        pairs = find_conflicts(g)
        used = [x for p in pairs for x in p]
        if len(used) == len(set(used)) and len(pairs) >= 4:          # >=4 disjoint conflicts
            return puz, wrong
    raise AssertionError("no instance found")


def test_impossible_3_four_disjoint_conflicts_need_more_than_three_corrections():
    puz, wrong = _four_disjoint_conflicts(5)
    rd = synthetic_reading(puz, wrong, wrong_conf=(0.9, 0.08))
    for strat in ("ucs", "astar", "greedy"):
        res = run_pipeline(rd, strat)
        assert res.status == "unreadable", strat
        assert "NO SOLUTION" in res.message and "unreadable" in res.message
        assert len(res.least_confident) == 5 and res.solution is None


def test_impossible_4_broken_reading_with_corrections_switched_off():
    puz, sol = make_case(9)
    rng = random.Random(9)
    c = rng.choice([i for i, v in enumerate(puz) if v])
    rd = synthetic_reading(puz, {c: conflict_wrong_digit(puz, c, rng)})
    res = run_pipeline(rd, use_nn_correction=False)
    assert res.status == "unreadable" and "corrections are switched off" in res.message
    assert res.ac3_conflict.startswith("AC-3: domain of cell")
