"""Search problem: which digits of the first reading must be corrected?

State  : tuple of (cell, new_digit) pairs, sorted by cell (a set of corrections;
         generating cells in increasing order creates every set exactly once).
Action : correct one more given cell (index larger than the last corrected one) to one of
         its top ALTS alternative digits.
Cost   : log p(read) - log p(new)  >= 0   (read digit is the network's arg-max).
Goal   : the corrected puzzle is solvable (optionally: has exactly one solution).
Limit  : at most max_corrections corrections.

Heuristic (A* and Greedy): every duplicate-digit conflict must lose at least one of its two
cells.  For one conflict (a, b) the cheapest possible repair is the cheapest single
correction among its not-yet-corrected cells; h(state) = max over all current conflicts.
  * admissible : any goal below the state fixes every conflict, so it pays at least the
                 repair cost of the most expensive conflict
  * consistent : an edge that corrects a cell of conflict P costs >= P's repair bound;
                 an edge on another cell leaves P (and its bound) unchanged
  * h = inf when a conflict has no correctable cell left, or conflicts remain and the
    correction budget is used up.
"""
import math
import numpy as np
from src.csp.sudoku_csp import find_conflicts, solve, Counters

INF = float("inf")


def build_candidates(grid, probs, alts=2, cells=None):
    """cand[cell] = [(digit, cost), ...] cheapest first, for every given cell."""
    cand = {}
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    for i, v in enumerate(grid):
        if v == 0 or (cells is not None and i not in cells):
            continue
        r = v - 1
        order = [d for d in np.argsort(-probs[i]) if d != r][:alts]
        cand[i] = sorted(((int(d) + 1, max(0.0, float(logp[i, r] - logp[i, d]))) for d in order),
                         key=lambda t: t[1])
    return cand

# Search problem: each state represents a set of corrections made to the original Sudoku reading.
class CorrectionProblem:
    def __init__(self, grid, probs, max_corrections=3, alts=2, require_unique=False,
                 candidate_cells=None):
        self.base = list(grid)
        self.max_corrections = max_corrections
        self.require_unique = require_unique
        self.cand = build_candidates(grid, probs, alts, candidate_cells)
        self.cells = sorted(self.cand)
        self.min_cost = {c: self.cand[c][0][1] for c in self.cells}
        self.cache = {}
        self.solutions = {}
        self.stats = dict(csp_calls=0, quick_rejects=0, csp_assignments=0, csp_backtracks=0,
                          ac3_revisions=0)
        self.csp_counters = Counters()

    # ---- helpers -------------------------------------------------------
    def apply(self, state):
        g = self.base[:]
        for c, d in state:
            g[c] = d
        return g

    def _info(self, state):
        info = self.cache.get(state)
        if info is None:
            g = self.apply(state)
            info = (g, find_conflicts(g))
            self.cache[state] = info
        return info

    # ---- search interface ----------------------------------------------
    def initial_state(self):
        return ()
    # Generate possible next corrections using alternative digits for uncertain cells.
    def successors(self, state):
        if len(state) >= self.max_corrections:
            return
        last = state[-1][0] if state else -1
        for c in self.cells:
            if c > last:
                for d, cost in self.cand[c]:
                    a = (c, d)
                    yield a, state + (a,), cost
    # Estimate the minimum additional correction cost needed to resolve existing conflicts.
    def heuristic(self, state):
        _, conflicts = self._info(state)
        if not conflicts:
            return 0.0
        if len(state) >= self.max_corrections:
            return INF
        done = {c for c, _ in state}
        h = 0.0
        for a, b in conflicts:
            m = min((self.min_cost[x] for x in (a, b) if x in self.min_cost and x not in done),
                    default=INF)
            if m == INF:
                return INF
            h = max(h, m)
        return h
    # Check whether the corrected puzzle has no conflicts and can be solved by the CSP solver.
    def is_goal(self, state):
        g, conflicts = self._info(state)
        self.cache.pop(state, None)
        if conflicts:
            self.stats["quick_rejects"] += 1
            return False
        self.stats["csp_calls"] += 1
        r = solve(g, use_fc=True, use_ac3=True, order="mrv", max_solutions=2 if self.require_unique else 1)
        self.csp_counters.add(r.counters)
        ok = r.status == "solved" and (not self.require_unique or len(r.solutions) == 1)
        if ok:
            self.solutions[state] = r.solution
        return ok
