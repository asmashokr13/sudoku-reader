"""Experiment 1: UCS vs Greedy vs A* (+ UCS with uniqueness goal) for 0-4 misreads."""
import sys, time, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULT_DIR
from src.eval.common import *
from src.integration.pipeline import reading_from_cache, run_pipeline

LEVELS = [0, 1, 2, 3, 4]
SEEDS, PUZZLES_PER_SEED, GIVENS = 5, 20, 30
METHODS = {"UCS": dict(strategy="ucs"), "Greedy": dict(strategy="greedy"), "A*": dict(strategy="astar"),
           "UCS+unique": dict(strategy="ucs", require_unique=True)}


def run():
    model, x, y, probs, pools = load_world()
    rows = []
    for s in range(SEEDS):
        inst = make_instances(pools, seed_for(1, s), PUZZLES_PER_SEED, GIVENS, LEVELS)
        for k in LEVELS:
            for pi, ins in enumerate(inst[k]):
                rd = reading_from_cache(ins, probs)
                costs = {}
                for name, kw in METHODS.items():
                    r = run_pipeline(rd, **kw)
                    rec = recovered(ins, r)
                    costs[name] = r.total_cost
                    rows.append(dict(seed=s, misreads=k, puzzle=pi, method=name, status=r.status,
                                     cost=r.total_cost, n_corrections=len(r.corrections),
                                     expanded=r.search["expanded"], generated=r.search["generated"],
                                     csp_calls=r.csp["csp_calls"], seconds=r.seconds, injected=json.dumps(ins.injected),
                                     correction_log=json.dumps(r.corrections), **rec))
                a, u = costs["A*"], costs["UCS"]
                assert (np.isinf(a) and np.isinf(u)) or abs(a - u) < 1e-9, "A* must match UCS cost"
        print(f"exp1 seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "exp1_raw.csv", index=False)
    # per-seed means, then mean/std over seeds
    per_seed = df.groupby(["method", "misreads", "seed"]).agg(
        expanded=("expanded", "mean"), generated=("generated", "mean"), csp_calls=("csp_calls", "mean"),
        solved=("solved", "mean"), exact_solution=("exact_solution", "mean"),
        exact_corrections=("exact_corrections", "mean"), seconds=("seconds", "mean")).reset_index()
    # cost only over solved instances
    cs = df[df.solved].groupby(["method", "misreads", "seed"]).cost.mean().rename("cost").reset_index()
    per_seed = per_seed.merge(cs, how="left", on=["method", "misreads", "seed"])
    summ = per_seed.groupby(["method", "misreads"]).agg(["mean", "std"]).drop(columns="seed")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(RESULT_DIR / "exp1_summary.csv", index=False)
    plot(summ)
    return summ


def plot(summ):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for name in METHODS:
        d = summ[summ.method == name]
        ax[0].errorbar(d.misreads, d.expanded_mean, d.expanded_std, marker="o", capsize=3, label=name)
        ax[1].errorbar(d.misreads, 100 * d.exact_solution_mean, 100 * d.exact_solution_std, marker="o", capsize=3, label=name)
        ax[2].errorbar(d.misreads, d.cost_mean, d.cost_std, marker="o", capsize=3, label=name)
    ax[0].set_yscale("symlog"); ax[0].set_title("Nodes expanded (mean ± std over 5 seeds)")
    ax[1].set_title("Exact-solution recovery rate (%)"); ax[2].set_title("Cost of the solution found (solved cases)")
    for a in ax:
        a.set_xlabel("injected misreads"); a.set_xticks(LEVELS); a.grid(alpha=0.3)
    ax[0].legend(); fig.tight_layout(); fig.savefig(RESULT_DIR / "exp1_search.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    t = time.time(); run(); print("exp1 total seconds", round(time.time() - t))
