"""Edge cases (5)."""
import numpy as np
from src.integration.pipeline import run_pipeline, Reading, read_grid
from src.data.mosaic import blank_image
from tests.helpers import *


def test_edge_1_already_correct_reading_needs_zero_corrections():
    puz, sol = make_case(21)
    res = run_pipeline(synthetic_reading(puz))
    assert res.status == "solved" and res.corrections == [] and res.total_cost == 0
    assert res.solution == sol and res.search["expanded"] == 0
    assert "without any correction" in res.message


def test_edge_2_completely_filled_valid_grid():
    sol = digits(CLASSIC_SOLUTION)
    res = run_pipeline(synthetic_reading(sol))
    assert res.status == "solved" and res.solution == sol and res.corrections == []


def test_edge_3_very_few_givens_is_solvable_but_not_unique():
    puz = [0] * 81
    puz[0], puz[10], puz[20], puz[30] = 1, 2, 3, 4
    res = run_pipeline(synthetic_reading(puz))
    assert res.status == "solved" and res.corrections == []
    assert run_pipeline(synthetic_reading(puz), require_unique=True).status == "unreadable"


def test_edge_4_empty_grid_all_blank_cells():
    blank = np.stack([blank_image(np.random.default_rng(i)) for i in range(81)])
    class Never:                                    # the network must never be called for blank cells
        def predict_proba(self, x): raise AssertionError("NN called on blank cell")
    rd = read_grid(blank, Never())
    assert all(rd.blank) and rd.grid == [0] * 81
    assert run_pipeline(rd).status == "solved"


def test_edge_5_misread_that_creates_no_conflict_is_reported_honestly():
    """A single digit change can leave a different solvable puzzle: the system accepts
    the first solvable reading, which may not be the writer's solution (limit of the method)."""
    found = False
    for seed in range(40):
        puz, sol = make_case(300 + seed, 30)
        rng = random.Random(seed)
        c = rng.choice([i for i, v in enumerate(puz) if v])
        wrong = rng.choice([d for d in range(1, 10) if d != puz[c]])
        rd = synthetic_reading(puz, {c: wrong}, seed)
        res = run_pipeline(rd)
        if res.status == "solved" and res.corrections == [] and res.solution != sol:
            found = True
            break
    assert found, "expected at least one silently-wrong-but-solvable case"
