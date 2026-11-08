"""Sudoku as a CSP: Backtracking, Forward Checking and AC-3 as independent switches.

Variables : 81 cells (index r*9+c).  Values: 1..9.  Givens have singleton domains.
Constraint: cells in the same row, column or 3x3 box must differ (binary != constraints).

Domains are bit masks (bit d-1 set <=> digit d still possible).

Counters (see Counters): assignments, backtracks, consistency_checks,
fc_prunings, ac3_revisions, ac3_prunings.
  * assignment  = a value placed on a variable after passing the consistency test
  * backtrack   = a placed value that is later undone because nothing below it worked
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, asdict

FULL = 0x1FF # Each cell's possible digits are stored in 9 bits (one bit per digit 1..9).
# 0x1FF is 111111111 in binary, which means all nine digits are still possible


def _build_geometry():
    units = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)])
    peers = [set() for _ in range(81)]
    for u in units:
        for a in u:
            for b in u:
                if a != b:
                    peers[a].add(b)
    return units, [sorted(p) for p in peers]


UNITS, PEERS = _build_geometry()


def digits_of(mask: int):
    return [d for d in range(1, 10) if mask >> (d - 1) & 1]


def rc(cell: int):
    return divmod(cell, 9)


@dataclass
class Counters:
    assignments: int = 0
    backtracks: int = 0
    consistency_checks: int = 0
    fc_prunings: int = 0
    ac3_revisions: int = 0
    ac3_prunings: int = 0

    def add(self, other: "Counters"):
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self):
        return asdict(self)


@dataclass
class SolveResult:
    status: str                      # "solved" | "no_solution" | "aborted"
    solutions: list = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)
    reason: str = ""
    seconds: float = 0.0
    domains_after_ac3: list | None = None

    @property
    def solution(self):
        return self.solutions[0] if self.solutions else None

    @property
    def message(self):
        if self.status == "solved":
            return f"solved ({len(self.solutions)} solution(s) found)"
        if self.status == "no_solution":
            return f"NO SOLUTION: {self.reason}"
        return f"ABORTED: {self.reason}"


def find_conflicts(grid):
    """All pairs of cells that hold the same digit inside one row/column/box."""
    pairs = set()
    for unit in UNITS:
        seen = {}
        for c in unit:
            v = grid[c]
            if v:
                if v in seen:
                    for o in seen[v]:
                        pairs.add((o, c))
                    seen[v].append(c)
                else:
                    seen[v] = [c]
    return sorted(pairs)


def is_valid_partial(grid) -> bool:
    return not find_conflicts(grid)


def ac3(dom, counters: Counters):
    """AC-3 on the != constraint graph. Modifies dom in place.

    Arc (i, j) revises D(i): a value v in D(i) has no support if D(j) == {v}.
    Returns (ok, wipeout_cell).  Two equal givens in one unit => both singletons
    {v}; revising one against the other empties it (wipe-out).
    """
    queue = deque((i, j) for i in range(81) for j in PEERS[i])
    while queue:
        i, j = queue.popleft()
        counters.ac3_revisions += 1
        dj = dom[j] # If neighbour j has only ONE possible digit, cell i is not allowed to use it.
        if dj & (dj - 1) == 0 and dom[i] & dj:      # D(j) is a singleton that is in D(i)
            dom[i] &= ~dj
            counters.ac3_prunings += 1
            if dom[i] == 0:
                return False, i
# Cell i just lost a digit, so its other neighbours must be checked again
            for k in PEERS[i]:
                if k != j:
                    queue.append((k, i))
    return True, None


class _Abort(Exception):
    pass


class _Solver:
    def __init__(self, use_fc, order, max_solutions, max_assignments, rng, counters):
        self.use_fc = use_fc
        self.order = order
        self.max_solutions = max_solutions
        self.max_assignments = max_assignments
        self.rng = rng
        self.c = counters
        self.solutions = []

    # ---- helpers -------------------------------------------------------
    def _legal(self, v, d):
        vals = self.values
        for p in PEERS[v]:
            if vals[p] == d:
                return False
        return True

    def _legal_mask(self, v, dom0):
        vals = self.values
        mask = dom0[v]
        for p in PEERS[v]:
            if vals[p]:
                mask &= ~(1 << (vals[p] - 1))
        self.c.consistency_checks += 1
        return mask

    def _select(self, dom):
        vals = self.values
        if self.order == "static":
            for i in range(81):
                if vals[i] == 0:
                    return i
            return -1
        # MRV: choose the empty cell with the FEWEST possible digits,
        best, best_n = -1, 99
        for i in range(81):
            if vals[i] == 0:
                n = (dom[i] if self.use_fc else self._legal_mask(i, dom)).bit_count()
                if n < best_n:
                    best, best_n = i, n
                    if n <= 1:
                        break
        return best

    # Forward Checking (FC): after placing digit d in cell v, remove d from the
    # possible digits of every neighbour. If a neighbour has no digits left,
    # this choice is a dead end, so return None right away.
    def _forward(self, dom, v, d):
        bit = 1 << (d - 1)
        nd = dom[:]
        nd[v] = bit
        for p in PEERS[v]:
            if nd[p] & bit:
                nd[p] &= ~bit
                self.c.fc_prunings += 1
                if nd[p] == 0:
                    return None
        return nd

    # ---- recursion -----------------------------------------------------
    def rec(self, dom):
        v = self._select(dom)
        if v < 0:
            self.solutions.append(self.values[:])
            return len(self.solutions) >= self.max_solutions
        cand = digits_of(dom[v] if self.use_fc else self._legal_mask(v, dom)
                         if self.order != "static" else dom[v])
        if self.rng is not None:
            self.rng.shuffle(cand)
        for d in cand:
            if not self.use_fc:
                self.c.consistency_checks += 1
                if not self._legal(v, d):
                    continue
            self.c.assignments += 1
            if self.max_assignments is not None and self.c.assignments > self.max_assignments:
                raise _Abort()
            self.values[v] = d
            if self.use_fc:
                nd = self._forward(dom, v, d)
                if nd is not None and self.rec(nd):
                    return True
            else:
                if self.rec(dom):
                    return True
# This digit led to a dead end, so undo it (empty the cell) and count one backtrack before trying the next digit.
            self.values[v] = 0
            self.c.backtracks += 1
        return False


def solve(grid, *, use_fc=False, use_ac3=False, order="static", max_solutions=1,
          max_assignments=None, rng=None) -> SolveResult:
    """Solve a Sudoku given as 81 ints (0 = blank).

    use_fc   : Forward Checking during search.
    use_ac3  : AC-3 before search (trims domains, detects conflicts).
    order    : "static" (row-major) or "mrv" (smallest domain first).
    """
    t0 = time.perf_counter()
    c = Counters()
    values = list(grid)
    if len(values) != 81 or any(not (0 <= v <= 9) for v in values):
        raise ValueError("grid must be 81 integers in 0..9")

    def done(status, reason="", sols=None, dom=None):
        return SolveResult(status, sols or [], c, reason, time.perf_counter() - t0, dom)

    dom = [FULL if v == 0 else 1 << (v - 1) for v in values]

    if use_ac3:
        ok, cell = ac3(dom, c)
        if not ok:
            r, cc = rc(cell)
            return done("no_solution",
                        f"AC-3 emptied the domain of cell (row {r + 1}, col {cc + 1}): "
                        f"two equal digits in one row/column/box", dom=dom)
    else:
        conf = find_conflicts(values)
        if conf:
            a, b = conf[0]
            return done("no_solution",
                        f"cells {rc(a)} and {rc(b)} hold the same digit {values[a]} in one unit "
                        f"(detected by direct scan, no AC-3)")
        if use_fc:                                   # initial forward checking from the givens
            for i in range(81):
                if values[i]:
                    bit = 1 << (values[i] - 1)
                    for p in PEERS[i]:
                        if values[p] == 0 and dom[p] & bit:
                            dom[p] &= ~bit
                            c.fc_prunings += 1
                            if dom[p] == 0:
                                return done("no_solution",
                                            f"forward checking emptied cell {rc(p)}")

    # blank cell with empty domain after pruning
    for i in range(81):
        if dom[i] == 0:
            return done("no_solution", f"cell {rc(i)} has no legal digit", dom=dom)

    s = _Solver(use_fc, order, max_solutions, max_assignments, rng, c)
    s.values = values
    try:
        s.rec(dom)
    except _Abort:
        return done("aborted", f"more than {max_assignments} assignments", s.solutions, dom)
    if s.solutions:
        return done("solved", sols=s.solutions, dom=dom)
    return done("no_solution", "search exhausted every possibility", dom=dom)


def count_solutions(grid, limit=2) -> int:
    r = solve(grid, use_fc=True, use_ac3=False, order="mrv", max_solutions=limit)
    return len(r.solutions)


CONFIGS = {
    "BT": dict(use_fc=False, use_ac3=False),
    "BT+FC": dict(use_fc=True, use_ac3=False),
    "BT+FC+AC3": dict(use_fc=True, use_ac3=True),
}
