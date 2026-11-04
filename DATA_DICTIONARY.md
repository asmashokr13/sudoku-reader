# Data dictionary

"Real" = comes unchanged from MNIST. "Generated" = produced by our code (seeded by `TEAM_SEED` in `config.py`, currently 7).

## 1. Prepared MNIST (`data/mnist_1to9.npz`, made by `make_data.py`)
| Array | Shape | Real / generated | Meaning |
|---|---|---|---|
| `x_train`, `x_val`, `x_test` | (N, 28, 28) uint8 | **real** | Handwritten digit images, pixel 0-255 (scaled to 0-1 when loaded). |
| `y_train`, `y_val`, `y_test` | (N,) uint8 | **real** | Digit label 1-9 (digit 0 removed). The network uses class = digit - 1. |

Sizes: train 45,093 / validation 8,984 / test 9,020. Split: the 60,000 official training images are shuffled with seed `TEAM_SEED*1000+1`, the first 50,000 are train and the last 10,000 validation, then digit 0 is dropped (so 50k/10k becomes 45,093/8,984). The test set is the official 10,000 images (9,020 after dropping 0). The test set is never used for training or for fitting the temperature.

Source files: the four official gzip files, verified by MD5 (see `data/manifest.json`). They are downloaded from a GitHub mirror because `yann.lecun.com` is often unreachable; checksums prove the content is the official MNIST.

## 2. Sudoku puzzle (generated, `src/data/sudoku_gen.py`)
| Field | Type | Generated how |
|---|---|---|
| `solution` | 81 ints 1-9 (row-major) | Full valid grid by randomised backtracking (forward checking + MRV, shuffled value order). |
| `puzzle` | 81 ints, 0 = blank | Cells are removed in random order; a removal is kept only if the puzzle still has exactly one solution (solution counter stops at 2), until the requested number of givens is reached. |

## 3. Mosaic instance (`src/data/mosaic.py`, class `Instance`)
| Field | Real / generated | Meaning |
|---|---|---|
| `img_idx` (81 ints) | generated choice of **real** test images | Index into `x_test` of the image shown in each cell; -1 = blank cell. |
| blank cell image | **generated** | Gaussian noise (sd 0.01) clipped to 0-1. A cell is "blank" if its mean brightness < 0.015 (`BLANK_THRESHOLD`). The rule is not learned; the thinnest real digit in MNIST has mean brightness 0.025 (checked in `make_data.py` and in a test). |
| `injected` (list of cells) | **generated** | Cells that hold a *hard* image. |
| hard image | real image, generated selection | A test image of the true digit that the trained network misreads while the true digit is still within its top-3. Rule documented in `ImagePools`. |
| clean image | real image, generated selection | A test image the network reads correctly; used for every non-injected given so the number of misreads equals the number injected. |
| `puzzle`, `solution` | generated | As above (ground truth, used only for scoring, never given to the solver). |

Hard-image pool sizes with the current network (digit 1..9): 11, 46, 15, 25, 23, 18, 27, 15, 16 (reuse across puzzles is expected). The pools depend on the trained network, so retraining changes them.

## 4. Network outputs
| Item | Meaning |
|---|---|
| `probs[cell, k]` | Temperature-scaled probability that the cell shows digit k+1 (zeros for blank cells). |
| `cost(cell, read -> new)` | `max(0, log p(read) - log p(new))`; read digit is the arg-max so it is never negative. |

## 5. Result files (`results/`)
| File | Columns |
|---|---|
| `exp1_raw.csv` | seed, misreads, puzzle, method, status, cost, n_corrections, expanded, generated, csp_calls, seconds, injected (cells), correction_log (JSON: cell,row,col,read,new,cost), solved, exact_solution, exact_corrections |
| `exp2_raw.csv` | seed, givens, puzzle, order, config, status, correct, seconds, assignments, backtracks, consistency_checks, fc_prunings, ac3_revisions, ac3_prunings |
| `exp3_raw.csv` | seed, misreads, puzzle, method, solved, exact_solution, candidate_sets_tested |
| `*_summary.csv`, `tables.md` | Mean and standard deviation over the 5 seeds (std of the per-seed means). |
| `nn_results.json` | Accuracy, NLL, ECE (raw / temperature scaled), confusion matrices for both classifiers. |

Metric definitions: **exact-solution recovery** = the solver's final grid equals the writer's original solution; **exact corrections** = the corrected cells and digits are exactly the injected misreads; **solved** = some solvable puzzle was found (possibly a different solution).
