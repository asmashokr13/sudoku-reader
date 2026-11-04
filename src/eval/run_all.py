"""python -m src.eval.run_all  : run the three experiments and build the tables."""
from . import exp1_search, exp2_csp, exp3_nn, make_tables

if __name__ == "__main__":
    exp1_search.run(); exp2_csp.run(); exp3_nn.run(); make_tables.main()
