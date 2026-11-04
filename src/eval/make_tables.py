"""Turn the experiment CSVs into markdown tables (mean ± std over 5 seeds): results/tables.md"""
import json
import pandas as pd
from config import RESULT_DIR


def pm(df, col, digits=2, pct=False):
    f = 100 if pct else 1
    return (df[f"{col}_mean"] * f).round(digits).astype(str) + " ± " + (df[f"{col}_std"] * f).round(digits).astype(str)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    parts = []
    nn = json.loads((RESULT_DIR / "nn_results.json").read_text())
    rows = []
    for name, key in (("MLP 784-128-64-9", "mlp"), ("Logistic regression", "logreg")):
        t = nn[key]["test"]
        rows.append([name, f"{100 * t['accuracy']:.2f}", f"{t['nll_raw']:.4f}", f"{t['ece_raw']:.4f}",
                     f"{t['nll_ts']:.4f}", f"{t['ece_ts']:.4f}", f"{t['T']:.3f}"])
    parts += ["## Classifier on the MNIST test set (digits 1-9)",
              md(pd.DataFrame(rows, columns=["model", "accuracy %", "NLL raw", "ECE raw", "NLL temp-scaled", "ECE temp-scaled", "T (fit on validation)"]))]

    s1 = pd.read_csv(RESULT_DIR / "exp1_summary.csv")
    t1 = pd.DataFrame({"method": s1.method, "misreads": s1.misreads,
                       "nodes expanded": pm(s1, "expanded", 1), "nodes generated": pm(s1, "generated", 1),
                       "CSP calls": pm(s1, "csp_calls", 1), "solved %": pm(s1, "solved", 1, True),
                       "exact-solution recovery %": pm(s1, "exact_solution", 1, True),
                       "cost (solved)": pm(s1, "cost", 2)})
    parts += ["## Experiment 1 - UCS vs Greedy vs A* (5 seeds x 20 puzzles = 100 puzzles per misread level)", md(t1)]

    s2 = pd.read_csv(RESULT_DIR / "exp2_summary.csv")
    t2 = pd.DataFrame({"order": s2.order, "config": s2.config, "givens": s2.givens,
                       "assignments": pm(s2, "assignments", 1), "backtracks": pm(s2, "backtracks", 1),
                       "seconds": pm(s2, "seconds", 4), "aborted (cap 300k) %": pm(s2, "aborted", 1, True)})
    parts += ["## Experiment 2 - BT vs BT+FC vs BT+FC+AC-3 (5 seeds x 10 puzzles per givens level)", md(t2)]

    s3 = pd.read_csv(RESULT_DIR / "exp3_summary.csv")
    t3 = pd.DataFrame({"method": s3.method, "misreads": s3.misreads, "solved %": pm(s3, "solved", 1, True),
                       "exact-solution recovery %": pm(s3, "exact_solution", 1, True),
                       "candidate sets tested": pm(s3, "candidate_sets_tested", 1)})
    parts += ["## Experiment 3 - NN-guided vs no correction vs random correction (5 seeds x 20 puzzles)", md(t3)]
    (RESULT_DIR / "tables.md").write_text("\n\n".join(parts) + "\n")
    print("wrote results/tables.md")


if __name__ == "__main__":
    main()
