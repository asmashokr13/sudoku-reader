"""Seeded Sudoku generator: full grid by randomised backtracking, then remove cells
while the solution stays unique."""
import random
from src.csp.sudoku_csp import solve, count_solutions


def random_solution(rng: random.Random):
    r = solve([0] * 81, use_fc=True, order="mrv", rng=rng)
    return r.solution


def make_puzzle(rng: random.Random, n_givens: int, max_restarts: int = 50):
    """Return (puzzle, solution). Puzzle has exactly n_givens givens and a unique solution."""
    for _ in range(max_restarts):
        sol = random_solution(rng)
        puz = sol[:]
        cells = list(range(81))
        rng.shuffle(cells)
        givens = 81
        for c in cells:
            if givens <= n_givens:
                break
            keep = puz[c]
            puz[c] = 0
            if count_solutions(puz, 2) == 1:
                givens -= 1
            else:
                puz[c] = keep
        if givens == n_givens:
            return puz, sol
    raise RuntimeError(f"could not reach {n_givens} givens with a unique solution")
