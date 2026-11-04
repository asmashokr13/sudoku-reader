"""Experiment 2: BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens."""
import random, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULT_DIR
from src.eval.common import seed_for
from src.data.sudoku_gen import make_puzzle
from src.csp.sudoku_csp import solve, CONFIGS

GIVENS = [25, 30, 35]
SEEDS, PUZZLES_PER_SEED = 5, 10
CAP = 300_000                                  # assignments before a run is declared "aborted"
ORDERS = ["static", "mrv"]


def run():
    rows = []
    for s in range(SEEDS):
        rng = random.Random(seed_for(2, s))
        for g in GIVENS:
            for pi in range(PUZZLES_PER_SEED):
                puz, sol = make_puzzle(rng, g)
                for order in ORDERS:
                    for name, cfg in CONFIGS.items():
                        r = solve(puz, order=order, max_assignments=CAP, **cfg)
                        ok = r.status == "solved" and r.solution == sol
                        rows.append(dict(seed=s, givens=g, puzzle=pi, order=order, config=name,
                                         status=r.status, correct=ok, seconds=r.seconds, **r.counters.as_dict()))
        print(f"exp2 seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "exp2_raw.csv", index=False)
    df["aborted"] = df.status == "aborted"
    per_seed = df.groupby(["order", "config", "givens", "seed"])[
        ["assignments", "backtracks", "seconds", "aborted", "consistency_checks", "fc_prunings", "ac3_revisions"]].mean().reset_index()
    summ = per_seed.groupby(["order", "config", "givens"]).agg(["mean", "std"])
    summ = summ.drop(columns="seed"); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index(); summ.to_csv(RESULT_DIR / "exp2_summary.csv", index=False)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for r_, order in enumerate(ORDERS):
        for c_, (metric, title) in enumerate((("assignments", "Assignments"), ("backtracks", "Backtracks"), ("seconds", "Seconds"))):
            a = ax[r_, c_]
            for i, name in enumerate(CONFIGS):
                d = summ[(summ.order == order) & (summ.config == name)]
                a.bar(np.arange(3) + 0.27 * (i - 1), d[f"{metric}_mean"], 0.27, yerr=d[f"{metric}_std"], capsize=2, label=name)
            a.set_yscale("log"); a.set_xticks(range(3)); a.set_xticklabels([f"{g} givens" for g in GIVENS])
            a.set_title(f"{title} ({order} order)"); a.grid(alpha=0.3, axis="y")
    ax[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULT_DIR / "exp2_csp.png", dpi=130); plt.close(fig)
    return summ


if __name__ == "__main__":
    t = time.time(); run(); print("exp2 total seconds", round(time.time() - t))
