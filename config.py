"""Global configuration. Every random choice in the project derives from TEAM_SEED."""
from pathlib import Path

TEAM_SEED = 7                    # <- change to your team's own generator seed
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_DIR = ROOT / "results"

# Blank-cell rule: a 28x28 cell whose mean brightness is below this is "blank".
BLANK_THRESHOLD = 0.015
MAX_CORRECTIONS = 3              # search budget (project statement)
ALTS_PER_CELL = 2                # top alternatives considered for every given cell
