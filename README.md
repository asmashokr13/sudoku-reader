# Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

A neural network reads a 9x9 mosaic of handwritten digits. When it misreads a digit the puzzle
breaks. **Search** finds the cheapest set of corrections (at most 3), **CSP** (Backtracking,
Forward Checking, AC-3) checks whether each corrected puzzle is solvable and solves it.

```
mosaic -> blank rule + NN (probabilities) -> AC-3 diagnosis -> Search (UCS / Greedy / A*)
       -> CSP goal test on every proposed correction set -> solution + correction log
                                   \-> if nothing works within 3 corrections: "unreadable" + least confident cells
```

## 1. Environment
```
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # versions are pinned
```
Tested with Python 3.12.3 on CPU only (no GPU needed). The network is written in NumPy, so there is no PyTorch dependency.

## 2. Exact run commands
```
python make_data.py                  # download + verify MNIST, split 50k/10k, drop digit 0     (~1 min)
python -m src.nn.train               # MLP + logistic regression, calibration, plots             (~1.5 min)
python -m pytest tests -q            # 19 tests: 10 normal, 5 edge, 4 impossible                  (~5 s)
python run_demo.py                   # integrated demo, plot in results/demo/demo_overview.png
python run_demo.py --misreads 4      # an unreadable puzzle (needs 4 > 3 corrections)
python run_demo.py --strategy astar --unique    # A* with the "exactly one solution" goal
python -m src.eval.run_all           # the 3 experiments (5 seeds each) + results/tables.md     (~4 min)
```
All randomness comes from `TEAM_SEED` in `config.py` (currently 7). Change it to your team's seed and re-run everything.

## 3. Layout
| Path | Content |
|---|---|
| `make_data.py`, `src/data/` | MNIST download/split, Sudoku generator, mosaic builder, hard-digit injection |
| `src/nn/` | NumPy MLP, calibration (ECE, temperature scaling), training script, logistic-regression baseline |
| `src/csp/sudoku_csp.py` | BT / FC / AC-3 as independent switches (`use_fc`, `use_ac3`, `order`), counters |
| `src/search/` | generic UCS / Greedy / A* with counters (`best_first.py`), correction problem + heuristic (`correction.py`) |
| `src/integration/pipeline.py` | NN -> AC-3 -> Search -> CSP, correction log, "unreadable" report |
| `src/eval/` | experiments 1-3, tables, run_all |
| `tests/` | normal / edge / impossible cases, Romania map with known optimal cost 418 |
| `DATA_DICTIONARY.md` | every column / array, real vs generated |

## 4. Design decisions (short)
* **State** = set of (cell, new digit) corrections, generated in increasing cell order so every set appears once. **Cost** = `log p(read) - log p(new)` >= 0. **Goal** = corrected puzzle solvable (optionally exactly one solution). **Budget** = 3 corrections; candidates = every given cell x its 2 best alternatives.
* **A\* heuristic** = max over current duplicate-digit conflicts of the cheapest single repair of that conflict (infinite if no repair is possible). It is admissible and consistent (argument in `correction.py`, checked by `test_normal_7`); A* and UCS return the same cost in all 500 experiment instances.
* **Goal test** applies a quick duplicate scan first, then the CSP (FC + AC-3 + MRV). `quick_rejects` and `csp_calls` are counted separately from search nodes.
* **Injected misreads**: real MNIST test images that the network gets wrong while the true digit stays in its top-3; all other givens use correctly read images. Hence the number of misreads is exactly controlled.

## 5. Results (details in `results/tables.md`, plots `results/exp*.png`)
All numbers are mean over 5 seeds (seed 7, team seed set in `config.py`); see `results/tables.md` for the standard deviations. Main observations:
1. The MLP reaches 97.65 % test accuracy against 92.25 % for logistic regression; temperature scaling (T = 1.307) lowers its ECE from 0.0059 to 0.0043.
2. A* and UCS always find the same (optimal) cost; A* expands far fewer nodes (e.g. ~714 vs ~13,590 at 4 misreads, 7.6 vs 44.8 at 3 misreads). Greedy expands fewer nodes than UCS but is not optimal, so it recovers fewer exact solutions (59 % vs 72 % at 2 misreads, 38 % vs 49 % at 3).
3. With a *row-major* variable order FC and AC-3 reduce assignments by large factors (25 givens: ~47,500 -> ~10,900 -> ~7,000). With the MRV order FC and FC+AC-3 are identical, because MRV already exploits the singleton domains AC-3 would create.
4. Recovery is not 100 % even for one misread (82 %): a single wrong digit often leaves a *different* solvable puzzle, and "solvable" is accepted first. The uniqueness goal raises recovery (one misread: 82 % -> 99 %; two: 72 % -> 88 %; three: 49 % -> 86 %).
5. NN-guided correction clearly beats the alternatives: without correction only 18 % / 2 % / 0 % of the puzzles with 1 / 2 / 3 misreads can be solved at all, and random correction recovers 12 % / 0 % / 0 % of the exact solutions, against 79 % / 71 % / 68 % for the NN-guided search.
6. Temperature scaling does not change which corrections are chosen, because every cost is a difference of logits divided by the same T; it only rescales the costs (calibration still matters for reading the cost as a probability).
7. At 4 misreads only 3 corrections are allowed, so exact recovery is 0 by design.

## 6. Honest limitations
* Mosaics are built from isolated MNIST digits, not photos of real Sudoku grids; blank cells are synthetic noise.
* The hard-image pool is small (~200 images), so the same hard images recur across puzzles.
* The MLP is trained for 10 epochs only (97.65 % test accuracy); the logistic-regression baseline stopped at the 100-iteration limit (sklearn convergence warning), which is fine for a baseline but should be mentioned.
* Experiment 2 caps a run at 300,000 assignments; plain BT with a row-major order hit the cap in 4.0 % of the runs with 25 givens (2 of 50 runs, counted as aborted), so its true mean is higher than reported.

## 7. Checklist status
Done in this repository: items 1-10 of the common checklist (code, `make_data.py`, data dictionary, NN vs baseline, search with counters and known-answer test, CSP with switches and counters, `run_demo.py`, README, tests, experiments).
**Still to be produced by the team:** report (item 11), 10+5 minute presentation (item 12), individual contribution statements and the git log (item 13).
