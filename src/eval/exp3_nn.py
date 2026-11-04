"""Experiment 3: NN-guided correction vs no correction vs random correction."""
import random, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULT_DIR, MAX_CORRECTIONS
from src.eval.common import *
from src.integration.pipeline import reading_from_cache, run_pipeline, Reading
from src.csp.sudoku_csp import find_conflicts, solve

LEVELS = [0, 1, 2, 3]
SEEDS, PUZZLES_PER_SEED, GIVENS = 5, 20, 30
RANDOM_BUDGET = 200          # random correction sets tried before giving up


def random_correction(reading, rng, budget=RANDOM_BUDGET):
    """Try the first reading, then random correction sets (1-3 cells, random other digit)."""
    givens = [i for i, v in enumerate(reading.grid) if v]
    sets = [()] + [None] * budget
    for tries, _ in enumerate(sets):
        corr = ()
        if tries:
            n = rng.randint(1, MAX_CORRECTIONS)
            cells = rng.sample(givens, n)
            corr = tuple((c, rng.choice([d for d in range(1, 10) if d != reading.grid[c]])) for c in cells)
        g = reading.grid[:]
        for c, d in corr:
            g[c] = d
        if find_conflicts(g):
            continue
        r = solve(g, use_fc=True, use_ac3=True, order="mrv")
        if r.status == "solved":
            return r.solution, tries
    return None, budget


def run():
    model, x, y, probs, pools = load_world()
    raw_probs = model.predict_proba(x, calibrated=False)
    rows = []
    for s in range(SEEDS):
        inst = make_instances(pools, seed_for(3, s), PUZZLES_PER_SEED, GIVENS, LEVELS)
        rng = random.Random(seed_for(3, s))
        for k in LEVELS:
            for pi, ins in enumerate(inst[k]):
                rd = reading_from_cache(ins, probs)
                rd_raw = reading_from_cache(ins, raw_probs)
                out = {}
                r = run_pipeline(rd, "ucs"); out["NN-guided (calibrated)"] = (r.solution, r.search["goal_tests"])
                r = run_pipeline(rd_raw, "ucs"); out["NN-guided (raw probs)"] = (r.solution, r.search["goal_tests"])
                r = run_pipeline(rd, "ucs", use_nn_correction=False); out["No correction"] = (r.solution, 1)
                sol, tries = random_correction(rd, rng); out["Random correction"] = (sol, tries + 1)
                for name, (sol, tests) in out.items():
                    rows.append(dict(seed=s, misreads=k, puzzle=pi, method=name, solved=sol is not None,
                                     exact_solution=sol == ins.solution, candidate_sets_tested=tests))
        print(f"exp3 seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "exp3_raw.csv", index=False)
    per_seed = df.groupby(["method", "misreads", "seed"])[["solved", "exact_solution", "candidate_sets_tested"]].mean().reset_index()
    summ = per_seed.groupby(["method", "misreads"])[["solved", "exact_solution", "candidate_sets_tested"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index(); summ.to_csv(RESULT_DIR / "exp3_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for name in df.method.unique():
        d = summ[summ.method == name]
        ax[0].errorbar(d.misreads, 100 * d.exact_solution_mean, 100 * d.exact_solution_std, marker="o", capsize=3, label=name)
        ax[1].errorbar(d.misreads, 100 * d.solved_mean, 100 * d.solved_std, marker="o", capsize=3, label=name)
    ax[0].set_title("Exact-solution recovery (%)"); ax[1].set_title("Puzzle solved at all (%)")
    for a in ax:
        a.set_xlabel("injected misreads"); a.set_xticks(LEVELS); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULT_DIR / "exp3_nn.png", dpi=130); plt.close(fig)
    return summ


if __name__ == "__main__":
    t = time.time(); run(); print("exp3 total seconds", round(time.time() - t))
