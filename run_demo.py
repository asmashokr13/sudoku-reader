#!/usr/bin/env python
"""Integrated demo:  python run_demo.py [--seed 3] [--misreads 2] [--givens 30] [--strategy ucs]

NN reads a handwritten-digit mosaic -> AC-3 diagnoses -> Search proposes corrections ->
CSP tests each corrected puzzle.  Shows the input, the search trace, the solution, the
system with the NN switched on vs off, CSP pruning with/without FC and AC-3, and saves plots.
"""
import argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULT_DIR, ARTIFACT_DIR
from src.eval.common import load_world
from src.data.sudoku_gen import make_puzzle
from src.data.mosaic import build_instance
from src.integration.pipeline import read_grid, run_pipeline
from src.csp.sudoku_csp import solve, CONFIGS, ac3, Counters, FULL, rc


def grid_text(g, mark=None):
    """Text Sudoku; a digit followed by * is a marked cell."""
    mark = mark or {}
    lines = []
    for r in range(9):
        if r % 3 == 0:
            lines.append("  +---------+---------+---------+")
        row = "  "
        for c in range(9):
            if c % 3 == 0:
                row += "|"
            v = g[r * 9 + c]
            row += (" " + ("." if v == 0 else str(v)) + ("*" if (r * 9 + c) in mark else " "))
        lines.append(row + "|")
    lines.append("  +---------+---------+---------+")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--strategy", default="ucs", choices=["ucs", "greedy", "astar"])
    ap.add_argument("--unique", action="store_true", help="goal = exactly one solution")
    a = ap.parse_args()
    if not (ARTIFACT_DIR / "mlp.npz").exists():
        raise SystemExit("Trained network missing. Run:  python make_data.py && python -m src.nn.train")
    out = RESULT_DIR / "demo"; out.mkdir(parents=True, exist_ok=True)

    model, x, y, probs, pools = load_world()
    puz, sol = make_puzzle(random.Random(a.seed), a.givens)
    ins = build_instance(puz, sol, pools, a.misreads, np.random.default_rng(a.seed), a.seed)
    images = ins.images(x)                                   # the 9x9 mosaic of 28x28 images
    rd = read_grid(images, model)                            # blank rule + NN on the real pixels

    print("=" * 70, "\nINPUT: 9x9 mosaic of handwritten digits (saved to demo_overview.png)")
    print(f"ground-truth puzzle ({a.givens} givens, unique solution), digits marked * are the ones the NN will misread:")
    print(grid_text(puz, {c: 1 for c in ins.injected}))
    print("\nFIRST READING by the network:")
    wrongly = {c for c in range(81) if rd.grid[c] != puz[c]}
    print(grid_text(rd.grid, {c: 1 for c in wrongly}))
    for c in sorted(wrongly):
        top = np.argsort(-rd.probs[c])[:3]
        print(f"  cell (r{rc(c)[0]+1},c{rc(c)[1]+1}): read {rd.grid[c]} "
              f"(true {puz[c]}); top-3: " + ", ".join(f"{t+1}:{rd.probs[c][t]:.2f}" for t in top))

    res = run_pipeline(rd, a.strategy, require_unique=a.unique, trace_limit=12)
    print("\n" + "=" * 70, f"\nPIPELINE ({a.strategy.upper()}, NN ON)\n  {res.ac3_conflict}")
    print("  search trace (first expansions: #, cost so far g, heuristic h, corrections):")
    for n, g, h, s in res.trace:
        print(f"    #{n:<3} g={g:6.3f} h={h:6.3f}  {[(f'r{rc(c)[0]+1}c{rc(c)[1]+1}', d) for c, d in s]}")
    print(f"  search counters: {res.search}\n  CSP counters   : {res.csp}")
    print(f"  RESULT: {res.message}")
    if res.corrections:
        print("  correction log (cell, read digit, new digit, cost):")
        for c in res.corrections:
            print(f"    (r{c['row']},c{c['col']})  {c['read']} -> {c['new']}   cost {c['cost']}")
    if res.solution:
        print("  solution:"); print(grid_text(res.solution))
        print("  matches the writer's original solution:", res.solution == sol)

    off = run_pipeline(rd, a.strategy, use_nn_correction=False)
    print("\n" + "=" * 70, "\nNN-GUIDED CORRECTION OFF (solve the first reading as it is)\n  RESULT:", off.message[:160])
    if off.solution:
        print("  solved, but equals the original solution:", off.solution == sol)

    # --- strategies side by side -------------------------------------------
    strat = {s: run_pipeline(rd, s) for s in ("ucs", "greedy", "astar")}
    print("\n" + "=" * 70, "\nSEARCH STRATEGIES (nodes generated / expanded, CSP calls, total cost)")
    for s, r in strat.items():
        print(f"  {s:7s} generated {r.search['generated']:6d}  expanded {r.search['expanded']:6d}  "
              f"CSP calls {r.csp['csp_calls']:4d}  cost {r.total_cost:.3f}")

    # --- CSP with / without FC and AC-3 on the corrected puzzle --------------
    if res.solution:
        fixed = rd.grid[:]
        for k in res.corrections:
            fixed[k["cell"]] = k["new"]
    else:
        fixed = puz[:]
    csp = {n: solve(fixed, order="static", **cfg) for n, cfg in CONFIGS.items()}
    print("\n" + "=" * 70, "\nCSP PRUNING on the corrected puzzle (row-major order)")
    for n, r in csp.items():
        c = r.counters
        print(f"  {n:10s} assignments {c.assignments:6d}  backtracks {c.backtracks:6d}  "
              f"FC prunings {c.fc_prunings:6d}  AC-3 prunings {c.ac3_prunings:4d}")

    # --- plots ---------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 3, 1)
    rows = [np.hstack(images[r * 9:(r + 1) * 9]) for r in range(9)]
    ax.imshow(np.vstack(rows), cmap="gray_r", vmin=0, vmax=1); ax.set_title("INPUT: handwritten mosaic")
    for i in range(10):
        ax.axhline(i * 28 - 0.5, color="C0", lw=1.4 if i % 3 == 0 else 0.3)
        ax.axvline(i * 28 - 0.5, color="C0", lw=1.4 if i % 3 == 0 else 0.3)
    for c in wrongly:
        r_, c_ = rc(c)
        ax.add_patch(plt.Rectangle((c_ * 28 - 0.5, r_ * 28 - 0.5), 28, 28, fill=False, ec="red", lw=2))
    for k in res.corrections:
        r_, c_ = k["row"] - 1, k["col"] - 1
        ax.add_patch(plt.Rectangle((c_ * 28 + 1.5, r_ * 28 + 1.5), 25, 25, fill=False, ec="limegreen", lw=2, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0, 262, "red = misread by NN, green dashed = corrected by search", fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    ax.axis("off"); ax.set_title("Reading -> corrections -> solution")
    table = [["cell", "read", "new", "cost"]] + [[f"r{k['row']}c{k['col']}", k["read"], k["new"], f"{k['cost']:.2f}"] for k in res.corrections] or [["-"] * 4]
    ax.table(cellText=table, loc="center", cellLoc="center").scale(1, 1.6)
    ax.text(0.5, 0.05, ("NN ON: " + res.status + " | NN OFF: " + off.status +
                        (" (wrong solution)" if off.solution and off.solution != sol else "")),
            ha="center", transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(2, 3, 3)
    names = list(strat)
    w = 0.27
    ax.bar(np.arange(3) - w, [strat[s].search["generated"] for s in names], w, label="generated")
    ax.bar(np.arange(3), [strat[s].search["expanded"] for s in names], w, label="expanded")
    ax.bar(np.arange(3) + w, [strat[s].csp["csp_calls"] for s in names], w, label="CSP calls")
    ax.set_xticks(range(3)); ax.set_xticklabels([s.upper() for s in names]); ax.set_yscale("symlog")
    ax.set_title("SEARCH: effort per strategy"); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    cn = list(csp)
    ax.bar(np.arange(3) - 0.2, [csp[n].counters.assignments for n in cn], 0.4, label="assignments")
    ax.bar(np.arange(3) + 0.2, [csp[n].counters.backtracks for n in cn], 0.4, label="backtracks")
    ax.set_xticks(range(3)); ax.set_xticklabels(cn); ax.set_yscale("symlog")
    ax.set_title("CSP: pruning with / without FC and AC-3"); ax.legend(fontsize=8)

    for k_, (g, title) in enumerate(((rd.grid, "domain sizes after AC-3: first reading"),
                                     (fixed, "domain sizes after AC-3: corrected puzzle"))):
        ax = fig.add_subplot(2, 3, 5 + k_)
        dom = [FULL if v == 0 else 1 << (v - 1) for v in g]
        ok, cell = ac3(dom, Counters())
        sizes = np.array([d.bit_count() for d in dom]).reshape(9, 9)
        ax.imshow(sizes, cmap="viridis_r", vmin=0, vmax=9)
        for r_ in range(9):
            for c_ in range(9):
                ax.text(c_, r_, sizes[r_, c_], ha="center", va="center", fontsize=8, color="white" if sizes[r_, c_] < 5 else "black")
        ax.set_title(title + ("" if ok else "\n(WIPE-OUT: contradiction)"), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out / "demo_overview.png", dpi=120); plt.close(fig)
    print("\nPlot saved to", out / "demo_overview.png")


if __name__ == "__main__":
    main()
