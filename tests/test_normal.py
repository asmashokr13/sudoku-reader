"""Normal cases (10)."""
import random, itertools
import numpy as np
import pytest
from config import BLANK_THRESHOLD
from src.csp.sudoku_csp import solve, ac3, Counters, FULL, CONFIGS, count_solutions
from src.search.best_first import best_first_search
from src.search.correction import CorrectionProblem
from src.integration.pipeline import run_pipeline
from src.data.sudoku_gen import make_puzzle
from src.data.mosaic import blank_image, is_blank
from src.nn.mlp import softmax
from src.nn import calibration as cal
from tests.helpers import *
from tests.romania import RomaniaProblem, ROMANIA_UCS_COST, ROMANIA_GREEDY_COST


def test_normal_1_csp_all_configs_find_known_solution():
    for order in ("static", "mrv"):
        for name, cfg in CONFIGS.items():
            r = solve(digits(CLASSIC), order=order, **cfg)
            assert r.status == "solved" and r.solution == digits(CLASSIC_SOLUTION), (name, order)


def test_normal_2_ac3_prunes_domains_and_solves_easy_puzzle_without_backtracking():
    dom = [FULL if v == 0 else 1 << (v - 1) for v in digits(CLASSIC)]
    before = sum(d.bit_count() for d in dom)
    ok, _ = ac3(dom, Counters())
    assert ok and sum(d.bit_count() for d in dom) < before
    r = solve(digits(CLASSIC), use_fc=True, use_ac3=True)
    assert r.counters.backtracks == 0 and r.counters.ac3_prunings > 0


def test_normal_3_forward_checking_needs_fewer_assignments_than_plain_backtracking():
    puz, sol = make_case(5, 28)
    bt = solve(puz, use_fc=False, use_ac3=False)
    fc = solve(puz, use_fc=True, use_ac3=False)
    ac = solve(puz, use_fc=True, use_ac3=True)
    assert bt.solution == fc.solution == ac.solution == sol
    assert fc.counters.assignments < bt.counters.assignments
    assert ac.counters.assignments <= fc.counters.assignments


def test_normal_4_search_known_answer_on_romania_map():
    ucs = best_first_search(RomaniaProblem(), "ucs")
    astar = best_first_search(RomaniaProblem(), "astar")
    greedy = best_first_search(RomaniaProblem(), "greedy")
    assert ucs.cost == ROMANIA_UCS_COST == astar.cost
    assert ucs.state == "Bucharest" and ucs.path == ["Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"]
    assert greedy.cost == ROMANIA_GREEDY_COST > ucs.cost           # greedy is not optimal
    assert astar.expanded < ucs.expanded                           # heuristic saves work
    assert ucs.generated >= ucs.expanded > 0                       # counters are populated


def test_normal_5_one_misread_is_corrected_to_exact_solution_with_expected_cost():
    puz, sol = make_case(11)
    rng = random.Random(1)
    cell = rng.choice([i for i, v in enumerate(puz) if v])
    wrong = conflict_wrong_digit(puz, cell, rng)
    rd = synthetic_reading(puz, {cell: wrong})
    assert rd.grid[cell] == wrong
    for strat in ("ucs", "astar"):
        res = run_pipeline(rd, strat)
        assert res.status == "solved" and res.solution == sol
        assert [(c["cell"], c["read"], c["new"]) for c in res.corrections] == [(cell, wrong, puz[cell])]
        assert res.total_cost == pytest.approx(np.log(0.52 / 0.45), abs=1e-3)


def test_normal_6_ucs_equals_bruteforce_and_astar_equals_ucs():
    for seed in range(6):
        puz, sol = make_case(100 + seed, 32)
        rng = random.Random(seed)
        givens = [i for i, v in enumerate(puz) if v]
        bad = rng.sample(givens, 2)
        wrong = {c: conflict_wrong_digit(puz, c, rng) for c in bad}
        rd = synthetic_reading(puz, wrong, seed)
        cells = set(bad) | set(rng.sample([g for g in givens if g not in bad], 8))   # 10 candidate cells
        costs = {}
        for strat in ("ucs", "astar"):
            p = CorrectionProblem(rd.grid, rd.probs, 3, 2, candidate_cells=cells)
            costs[strat] = best_first_search(p, strat).cost
        # brute force over every correction set of the same candidate cells
        p = CorrectionProblem(rd.grid, rd.probs, 3, 2, candidate_cells=cells)
        best = float("inf")
        stack = [p.initial_state()]
        while stack:
            s = stack.pop()
            if p.is_goal(s):
                best = min(best, sum(dict(p.cand[c])[d] for c, d in s))
            stack.extend(n for _, n, _ in p.successors(s))
        assert costs["ucs"] == pytest.approx(best) == pytest.approx(costs["astar"])


def test_normal_7_heuristic_is_admissible_and_consistent():
    puz, sol = make_case(7, 30)
    rng = random.Random(3)
    givens = [i for i, v in enumerate(puz) if v]
    bad = rng.sample(givens, 2)
    rd = synthetic_reading(puz, {c: conflict_wrong_digit(puz, c, rng) for c in bad}, 3)
    cells = set(bad) | set(rng.sample([g for g in givens if g not in bad], 7))
    p = CorrectionProblem(rd.grid, rd.probs, 3, 2, candidate_cells=cells)

    def true_cost(state):                       # exact remaining cost inside the restricted graph
        best = float("inf")
        stack = [(state, 0.0)]
        while stack:
            s, g = stack.pop()
            if p.is_goal(s):
                best = min(best, g)
            stack.extend((n, g + c) for _, n, c in p.successors(s))
        return best

    states, frontier = [()], [()]
    for _ in range(2):
        frontier = [n for s in frontier for _, n, _ in p.successors(s)][:40]
        states += frontier
    for s in states:
        h = p.heuristic(s)
        assert h <= true_cost(s) + 1e-9                                  # admissible
        for _, n, c in p.successors(s):
            assert h <= c + p.heuristic(n) + 1e-9                        # consistent


def test_normal_8_generator_is_seeded_unique_and_valid():
    a, sa = make_puzzle(random.Random(42), 30)
    b, sb = make_puzzle(random.Random(42), 30)
    c, _ = make_puzzle(random.Random(43), 30)
    assert a == b and sa == sb and a != c
    assert sum(1 for v in a if v) == 30 and count_solutions(a, 5) == 1


def test_normal_9_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 4000)
    z = rng.normal(0, 1, (4000, 9)) * 2; z[np.arange(4000), y] += 3.0            # informative logits
    flip = rng.random(4000) < 0.15
    y_noisy = np.where(flip, rng.integers(0, 9, 4000), y)                       # makes net over-confident
    z = z * 3
    T = cal.fit_temperature(z, y_noisy)
    assert T > 1.5
    assert cal.nll(softmax(z / T), y_noisy) < cal.nll(softmax(z), y_noisy)
    assert cal.ece(softmax(z / T), y_noisy) < cal.ece(softmax(z), y_noisy)


def test_normal_10_blank_rule_separates_blank_cells_from_real_digits():
    from src.data.mnist import load_splits
    x = load_splits()["x_test"].astype(np.float32) / 255
    rng = np.random.default_rng(0)
    assert all(is_blank(blank_image(rng)) for _ in range(500))
    assert not any(is_blank(im) for im in x)                                   # no real digit looks blank
