"""NN reads the grid -> AC-3 diagnoses -> Search proposes corrections -> CSP tests them."""
from dataclasses import dataclass, field
import numpy as np
from config import MAX_CORRECTIONS, ALTS_PER_CELL
from src.data.mosaic import is_blank
from src.csp.sudoku_csp import ac3, Counters, FULL, rc, find_conflicts, solve
from src.search.best_first import best_first_search
from src.search.correction import CorrectionProblem


@dataclass
class Reading:
    grid: list                 # first reading, 0 = blank
    probs: np.ndarray          # (81, 9) probabilities (zeros for blank cells)
    blank: list                # bool per cell

    def confidence(self, cell):
        return float(self.probs[cell].max())


def read_grid(images, model) -> Reading:
    """images: (81,28,28) floats in 0..1. Blank cells by brightness rule, digits by the NN."""
    blank = [is_blank(im) for im in images]
    probs = np.zeros((81, 9), np.float32)
    idx = [i for i in range(81) if not blank[i]]
    if idx:
        probs[idx] = model.predict_proba(np.stack([images[i] for i in idx]))
    grid = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    return Reading(grid, probs, blank)


def reading_from_cache(instance, test_probs) -> Reading:
    """Same as read_grid but uses pre-computed network outputs (identical numbers, faster)."""
    probs = np.zeros((81, 9), np.float32)
    blank = [int(i) < 0 for i in instance.img_idx]
    for c, i in enumerate(instance.img_idx):
        if i >= 0:
            probs[c] = test_probs[i]
    grid = [0 if blank[c] else int(probs[c].argmax()) + 1 for c in range(81)]
    return Reading(grid, probs, blank)


@dataclass
class PipelineResult:
    status: str                         # "solved" | "unreadable"
    solution: list | None
    corrections: list                   # dicts: cell,row,col,read,new,cost
    total_cost: float
    ac3_conflict: str                   # what AC-3 said about the first reading
    search: dict                        # counters of the search
    csp: dict                           # aggregated CSP counters
    message: str
    least_confident: list = field(default_factory=list)
    strategy: str = ""
    trace: list = field(default_factory=list)
    seconds: float = 0.0


def least_confident_cells(reading: Reading, n=5):
    cells = [c for c in range(81) if not reading.blank[c]]
    cells.sort(key=reading.confidence)
    return [dict(cell=c, row=rc(c)[0] + 1, col=rc(c)[1] + 1, read=reading.grid[c],
                 confidence=round(reading.confidence(c), 4)) for c in cells[:n]]


def ac3_diagnosis(grid):
    dom = [FULL if v == 0 else 1 << (v - 1) for v in grid]
    ok, cell = ac3(dom, Counters())
    if ok:
        return "AC-3: no conflict found in the first reading"
    r, c = rc(cell)
    return f"AC-3: domain of cell (row {r + 1}, col {c + 1}) became empty -> the reading is inconsistent"


def run_pipeline(reading: Reading, strategy="ucs", max_corrections=MAX_CORRECTIONS,
                 alts=ALTS_PER_CELL, require_unique=False, use_nn_correction=True,
                 max_expansions=300_000, trace_limit=0, candidate_cells=None) -> PipelineResult:
    diag = ac3_diagnosis(reading.grid)
    prob = CorrectionProblem(reading.grid, reading.probs, max_corrections if use_nn_correction else 0,
                             alts, require_unique, candidate_cells)
    sr = best_first_search(prob, strategy, max_expansions, trace_limit)
    csp = prob.csp_counters.as_dict() | {k: prob.stats[k] for k in ("csp_calls", "quick_rejects")}
    search = sr.counters()
    if sr.found:
        grid = prob.apply(sr.state)
        corrections, total = [], 0.0
        for cell, new in sr.state:
            cost = dict(prob.cand[cell])[new]
            total += cost
            corrections.append(dict(cell=cell, row=rc(cell)[0] + 1, col=rc(cell)[1] + 1,
                                    read=reading.grid[cell], new=new, cost=round(cost, 4)))
        msg = ("solved without any correction" if not corrections
               else f"solved after {len(corrections)} correction(s), total cost {total:.3f}")
        return PipelineResult("solved", prob.solutions[sr.state], corrections, total, diag, search, csp,
                              msg, strategy=strategy, trace=sr.trace, seconds=sr.seconds)
    why = "search budget exhausted" if sr.aborted else (
        "no solvable puzzle within the correction budget" if use_nn_correction
        else "the first reading is not solvable and corrections are switched off")
    lc = least_confident_cells(reading)
    msg = (f"NO SOLUTION: puzzle unreadable ({why}). Least confident cells: " +
           ", ".join(f"(r{x['row']},c{x['col']}) read {x['read']} @ {x['confidence']:.2f}" for x in lc))
    return PipelineResult("unreadable", None, [], float("inf"), diag, search, csp, msg, lc, strategy,
                          sr.trace, sr.seconds)
